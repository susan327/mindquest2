import os
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo  # JST用
from difflib import SequenceMatcher

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    make_response,
    session,
    flash,
    jsonify,
    g,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# =========================
# ベースパス & .env
# =========================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# =========================
# Gemini 設定
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_TEXT = os.getenv("GEMINI_MODEL_TEXT", "gemini-2.5-flash-lite")
GEMINI_MODEL_JSON = os.getenv("GEMINI_MODEL_JSON", "gemini-2.5-flash-lite")

gemini_available = False
try:
    import google.generativeai as genai

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_available = True
except Exception as e:
    print("Gemini init error:", e)
    gemini_available = False


def gemini_generate_text(
    prompt,
    model_name=None,
    temperature=0.6,
    max_tokens=1024,
):
    """シンプルなテキスト生成ヘルパー"""
    if not gemini_available:
        return None
    model_name = model_name or GEMINI_MODEL_TEXT
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return (resp.text or "").strip()
    except Exception as e:
        print("Gemini text error:", e)
        return None


def gemini_generate_json(
    prompt,
    model_name=None,
    temperature=0.2,
    max_tokens=768,
):
    """JSONを返したいとき用。```json ... ``` を優先してパース。"""
    if not gemini_available:
        return None
    model_name = model_name or GEMINI_MODEL_JSON
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        txt = (resp.text or "").strip()

        import re
        import json as _json

        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, flags=re.S)
        raw = m.group(1) if m else txt
        return _json.loads(raw)
    except Exception as e:
        print("Gemini json error:", e)
        return None


# =========================
# Flask & DB 設定（Neon 前提）
# =========================
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "False") == "True"

    raw_url = os.getenv("DATABASE_URL", "")
    if not raw_url:
        raise RuntimeError("DATABASE_URL が設定されていません（Neon の URL を .env に入れてね）")

    SQLALCHEMY_DATABASE_URI = raw_url.replace("postgres://", "postgresql://")
    SQLALCHEMY_TRACK_MODIFICATIONS = False


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# =========================
# JST タイムゾーン & Jinjaフィルター
# =========================
JST = ZoneInfo("Asia/Tokyo")


@app.template_filter("jst")
def jst_filter(dt, fmt=None):
    """
    DB上はUTC想定のdatetimeを、日本時間に変換して表示用文字列にするフィルター。
    テンプレでは {{ entry.created_at|jst("%Y-%m-%d %H:%M") }} のように使う。
    """
    if dt is None:
        return ""
    # naiveならUTCとみなす
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(JST)
    if fmt:
        return dt.strftime(fmt)
    return dt


# =========================
# Models
# =========================
class User(db.Model):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    user_token = db.Column(db.String(64), unique=True, index=True, nullable=False)
    name = db.Column(db.String(32), nullable=False, default="冒険者")
    level = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # メール & パスワード（任意）
    email = db.Column(db.String(255), unique=True, index=True)
    password_hash = db.Column(db.String(255))


class DiagnosisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    top_type = db.Column(db.String(32))
    scores = db.Column(db.JSON, nullable=False, default=dict)      # 最終スコア
    raw_scores = db.Column(db.JSON)                                # 質問のみの素点
    bonus_scores = db.Column(db.JSON)                              # AI補正ぶん（0〜5）

    written1 = db.Column(db.Text)  # 最近よく考えること・悩み
    written2 = db.Column(db.Text)  # 日々の行動・習慣
    written3 = db.Column(db.Text)  # 理想の自分像


class Quest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=False)
    type_key = db.Column(db.String(32), nullable=False, default="common")
    category = db.Column(db.String(32), nullable=False, default="growth")
    structure = db.Column(db.String(32), default="single")  # "single" / "multi_step"
    steps_json = db.Column(db.JSON, default=list)           # ステップ構造のJSON（リスト想定）
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.Text)  # AIの感想を保存する


class KaiLog(db.Model):
    __tablename__ = "kai_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(128), nullable=False)  # 快の名前
    count = db.Column(db.Integer, default=0)          # 実行回数
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class QuestProgress(db.Model):
    """
    クエストの進捗:
    - not_started : 一度も開いていない or レコードがない
    - in_progress : 画面を開いた（挑戦中）
    - completed   : フィードバック送信まで完了
    """
    __tablename__ = "quest_progress"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    quest_id = db.Column(db.Integer, db.ForeignKey("quest.id"), nullable=False)
    status = db.Column(db.String(16), nullable=False, default="not_started")
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


def _similarity(a: str, b: str) -> float:
    """2つの文字列のざっくり類似度（0.0〜1.0）"""
    return SequenceMatcher(None, a, b).ratio()


def find_similar_kai(logs, name: str, threshold: float = 0.7):
    """
    既存の KaiLog 一覧の中から、
    name に「それっぽい」ものがあれば返す（なければ None）。
    threshold はどれくらい似てたら同一とみなすか（0.0〜1.0）。
    """
    name = (name or "").strip()
    if not name:
        return None

    best_log = None
    best_score = 0.0

    for log in logs:
        score = _similarity(log.name, name)
        if score > best_score:
            best_score = score
            best_log = log

    if best_log and best_score >= threshold:
        return best_log
    return None


# =========================
# Static data
# =========================
TYPE_KEYS = [
    "sage",
    "monk",
    "priest",
    "mage",
    "thief",
    "artist",
    "guardian",
    "commander",
]

TYPE_INFO = {
    "sage": {
        "name": "賢者",
        "feature": "思考深める分析家",
        "good": "洞察と計画性",
        "bad": "動きが遅くなりがち",
        "image": "sage.png",
    },
    "monk": {
        "name": "武闘家",
        "feature": "行動力と瞬発力",
        "good": "実行力とエネルギー",
        "bad": "衝動的になりがち",
        "image": "monk.png",
    },
    "priest": {
        "name": "僧侶",
        "feature": "思いやりと支え",
        "good": "共感とケア",
        "bad": "自分を後回しにしがち",
        "image": "priest.png",
    },
    "mage": {
        "name": "魔法使い",
        "feature": "創造と戦略",
        "good": "発想力と戦略性",
        "bad": "実務が苦手なときも",
        "image": "mage.png",
    },
    "thief": {
        "name": "盗賊",
        "feature": "柔軟な適応力",
        "good": "機転と探索力",
        "bad": "腰が落ち着かない",
        "image": "thief.png",
    },
    "artist": {
        "name": "芸術家",
        "feature": "感性の表現者",
        "good": "表現力",
        "bad": "ムラが出やすい",
        "image": "artist.png",
    },
    "guardian": {
        "name": "守護者",
        "feature": "堅実と信頼",
        "good": "安定感",
        "bad": "変化に慎重",
        "image": "guardian.png",
    },
    "commander": {
        "name": "指揮官",
        "feature": "リードと決断",
        "good": "リーダーシップ",
        "bad": "押しが強くなりがち",
        "image": "commander.png",
    },
    "common": {
        "name": "共通",
        "feature": "全タイプ共通",
        "good": "—",
        "bad": "—",
        "image": "sage.png",
    },
}

QUEST_TYPE_LABELS = {
    "growth": "成長",
    "communication": "コミュニケーション",
    "habits": "習慣",
    "action": "行動",
    "reflection": "内省",
    "self_understanding": "自己理解",
    "common": "全タイプ",
}

CHOICE_TO_SCORE = {
    "yes": 3,
    "maybe": 1,
    "neutral": 1,
    "no": 0,
}

QUESTIONS = [
    ["物事を分析しすぎて、決断が遅くなることがある"],
    ["計画を立ててからでないと動けないことが多い"],
    ["感情より理屈を優先してしまう傾向がある"],
    ["議論になると正しさを追求しすぎてしまう"],
    ["考えるより先に動いてしまうことがある"],
    ["衝動的に行動して、あとで振り返ることがある"],
    ["気持ちが高ぶるとつい強く出てしまうことがある"],
    ["ストレートな物言いで誤解されることがある"],
    ["相手のことを考えすぎて自分の意見を抑えてしまう"],
    ["困っている人を見ると手を差し伸べずにはいられない"],
    ["人の感情に敏感で、共感しすぎて疲れることがある"],
    ["誰かを傷つけないよう慎重に言葉を選ぶ"],
    ["気分によって考え方や意見が変わることがある"],
    ["その時の感情に任せて行動してしまうことがある"],
    ["気持ちの浮き沈みが激しいと感じる"],
    ["感情をうまく伝えるのが難しいと感じる"],
    ["自由で柔軟な発想を大切にしている"],
    ["ルールに縛られず、直感で動くことが多い"],
    ["自由を制限されるとストレスを感じる"],
    ["集団より一人で行動する方が気が楽"],
    ["独自の視点で物事を捉えるのが好きだ"],
    ["思いついたことをすぐ形にしたくなる"],
    ["感受性が強く、些細なことにも心が動く"],
    ["自分の世界を大切にしていて他人に踏み込まれたくない"],
    ["安定を求めて慎重に物事を考える"],
    ["リスクよりも確実性を優先する行動をとる"],
    ["大きな変化に対して不安を感じやすい"],
    ["協調性を大切にし、チームワークを重視する"],
    ["全体を俯瞰して効率よく進めることを考える"],
    ["自ら先頭に立って行動をリードすることが多い"],
    ["感情を抑えて冷静に振る舞おうとする傾向がある"],
    ["人を導いたり、指示を出す立場になることが多い"],
]

QUESTION_TYPES = [
    "sage",
    "sage",
    "sage",
    "sage",
    "monk",
    "monk",
    "monk",
    "monk",
    "priest",
    "priest",
    "priest",
    "priest",
    "mage",
    "mage",
    "mage",
    "mage",
    "thief",
    "thief",
    "thief",
    "thief",
    "artist",
    "artist",
    "artist",
    "artist",
    "guardian",
    "guardian",
    "guardian",
    "guardian",
    "commander",
    "commander",
    "commander",
    "commander",
]

# フロントから飛んでくる value を正規化して Quest.type_key に保存するためのマップ
RAW_TYPE_KEY_MAP = {
    "all": "common",
    "common": "common",
    "sage": "sage",
    "fighter": "monk",   # 武闘家 → monk
    "monk": "priest",    # 僧侶 → priest（古い値にも対応）
    "priest": "priest",
    "wizard": "mage",    # 魔法使い → mage
    "mage": "mage",
    "rogue": "thief",    # 盗賊 → thief
    "thief": "thief",
    "artist": "artist",
    "guardian": "guardian",
    "commander": "commander",
}


# =========================
# Helpers
# =========================
def get_or_set_user_token(resp=None):
    """
    user_token cookie を取得／発行。
    resp があればそのレスポンスに直接セット、
    なければ g.new_user_token に入れて after_request で付与。
    """
    token = request.cookies.get("user_token")
    if token and len(token) == 32:
        return token

    token = secrets.token_hex(16)

    if resp is not None:
        resp.set_cookie(
            "user_token",
            token,
            httponly=True,
            secure=app.config["SESSION_COOKIE_SECURE"],
            samesite="Lax",
            max_age=60 * 60 * 24 * 365,
        )
    else:
        g.new_user_token = token

    return token


@app.after_request
def apply_user_token_cookie(response):
    """g.new_user_token があれば cookie をセット。"""
    token = getattr(g, "new_user_token", None)
    if token and not request.cookies.get("user_token"):
        response.set_cookie(
            "user_token",
            token,
            httponly=True,
            secure=app.config["SESSION_COOKIE_SECURE"],
            samesite="Lax",
            max_age=60 * 60 * 24 * 365,
        )
    return response


def get_current_user():
    token = request.cookies.get("user_token")
    if not token:
        return None
    return User.query.filter_by(user_token=token).first()


def ensure_user():
    """
    どの画面から来ても：
    - cookie があればそのユーザー
    - なければ cookieを発行して「冒険者」を1件だけ作成
    """
    user = get_current_user()
    if user:
        return user

    token = get_or_set_user_token()
    user = User(user_token=token, name="冒険者", level=1)
    db.session.add(user)
    db.session.commit()
    return user


# =========================
# Routes: 基本
# =========================
@app.route("/")
def index():
    user = get_current_user()
    return render_template("index.html", user=user)


@app.route("/start")
def start():
    return render_template("start.html", questions=QUESTIONS)


# =========================
# 名前入力・変更
# =========================
@app.route("/name", methods=["GET", "POST"])
def name_input():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()[:10]
        if not name:
            flash("名前を入力してね")
            return redirect(url_for("name_input"))

        resp = make_response(redirect(url_for("menu")))
        token = get_or_set_user_token(resp)
        user = User.query.filter_by(user_token=token).first()
        if not user:
            user = User(user_token=token, name=name, level=1)
            db.session.add(user)
        else:
            user.name = name
        db.session.commit()
        return resp

    return render_template("name_input.html")


@app.route("/name/change", methods=["GET", "POST"])
def name_change():
    user = get_current_user()
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()[:10]
        if not name:
            flash("名前を入力してね")
            return redirect(url_for("name_change"))
        if user:
            user.name = name
            db.session.commit()
        session["display_name"] = name
        return redirect(url_for("index"))
    return render_template("name_change.html", current_name=user.name if user else None)


# =========================
# アカウント（メール＋パスワード）
# =========================
@app.route("/account", methods=["GET", "POST"])
def account():
    user = ensure_user()

    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = (request.form.get("password") or "").strip()
        password2 = (request.form.get("password2") or "").strip()

        # パスワード2回入力チェック
        if password != password2:
            flash("パスワードが一致しません。もう一度入力してください。", "account_error")
            return redirect(url_for("account"))

        # メール更新
        if email:
            user.email = email

        # パスワード更新（ハッシュ化）
        if password:
            user.password_hash = generate_password_hash(password)

        db.session.commit()
        flash("アカウント情報を保存しました", "account_success")
        return redirect(url_for("account"))

    return render_template("account.html", user=user)


@app.route("/account/delete", methods=["POST"])
def account_delete():
    """アカウント削除：関連データも消してログアウト"""
    user = get_current_user()
    if not user:
        flash("ログイン情報が見つかりませんでした")
        return redirect(url_for("index"))

    DiagnosisResult.query.filter_by(user_id=user.id).delete()
    JournalEntry.query.filter_by(user_id=user.id).delete()
    KaiLog.query.filter_by(user_id=user.id).delete()
    QuestProgress.query.filter_by(user_id=user.id).delete()
    db.session.commit()

    db.session.delete(user)
    db.session.commit()

    resp = make_response(redirect(url_for("index")))
    resp.delete_cookie("user_token")
    session.clear()
    flash("アカウントを削除しました")
    return resp


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = (request.form.get("password") or "").strip()

        user = User.query.filter_by(email=email).first()

        # ユーザーが存在しない or パスワード違う
        if not user or not check_password_hash(user.password_hash, password):
            return render_template(
                "login.html",
                error="メールアドレスまたはパスワードが違います。",
            )

        # 成功
        session["user_id"] = user.id
        return redirect(url_for("menu"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    resp = make_response(redirect(url_for("index")))
    resp.delete_cookie("user_token")
    session.clear()
    flash("ログアウトしました")
    return resp


# =========================
# メニュー
# =========================
@app.route("/menu")
def menu():
    user = ensure_user()
    display_name = user.name or "冒険者"
    level = user.level or 1

    user_type = "まだ診断していません"
    last_result = (
        DiagnosisResult.query.filter_by(user_id=user.id)
        .order_by(DiagnosisResult.created_at.desc())
        .first()
    )
    if last_result and last_result.top_type:
        info = TYPE_INFO.get(last_result.top_type)
        if info and "name" in info:
            user_type = info["name"]
        else:
            user_type = last_result.top_type

    return render_template(
        "menu.html",
        user=user,
        name=display_name,
        user_type=user_type,
        level=level,
    )


# =========================
# 性格診断 結果（AI補正 +5点まで）
# =========================
@app.route("/result", methods=["GET", "POST"])
def result():
    user = ensure_user()

    # GET: 最新の結果を表示
    if request.method == "GET":
        last = (
            DiagnosisResult.query.filter_by(user_id=user.id)
            .order_by(DiagnosisResult.created_at.desc())
            .first()
        )
        if not last:
            return redirect(url_for("start"))

        info = TYPE_INFO.get(last.top_type, TYPE_INFO["common"])

        # 診断コメント用 AI（任意）
        if gemini_available:
            prompt = f"""
あなたは「RPG風性格診断」の解説AIです。

タイプ一覧:
{json.dumps(TYPE_INFO, ensure_ascii=False, indent=2)}

最終スコア(final_scores):
{json.dumps(last.scores, ensure_ascii=False, indent=2)}

素点(raw_scores):
{json.dumps(last.raw_scores or {}, ensure_ascii=False, indent=2)}

ボーナス得点(bonus_scores):
{json.dumps(last.bonus_scores or {}, ensure_ascii=False, indent=2)}

トップタイプ: {last.top_type}（{TYPE_INFO.get(last.top_type, {}).get("name", "")}）

自由記述:
- 最近よく考えることや悩み: {last.written1 or ""}
- 日々の行動や習慣: {last.written2 or ""}
- 理想の自分像: {last.written3 or ""}

この人が「自分を責めすぎず、少しラクになれる」ようなコメントを、
やさしい日本語で 120〜200文字くらいで書いてください。

・説教しない
・診断結果を押しつけない
・良いところを1〜2個だけそっと伝える
"""
            comment = gemini_generate_text(prompt) or "（AIコメントの生成に失敗しました）"
        else:
            comment = "GEMINI_API_KEY が設定されていないため、AIコメントはオフになっています。"

        # テンプレ側で「素点」「ボーナス」「最終スコア」を全部見せられるように渡す
        return render_template(
            "result.html",
            info=info,
            score=last.scores,
            raw_score=last.raw_scores or last.scores,
            bonus_score=last.bonus_scores or {k: 0 for k in last.scores.keys()},
            written1=last.written1 or "",
            written2=last.written2 or "",
            written3=last.written3 or "",
            comment=comment,
            type_info=TYPE_INFO,
            type_keys=TYPE_KEYS,
        )

    # POST: 新しい結果を保存
    # --------------------------------
    # ① 質問への回答から「素点(raw_scores)」を計算
    answers_raw = {}
    num_q = len(QUESTIONS)
    for i in range(num_q):
        answers_raw[f"q{i}"] = request.form.get(f"q{i}", "no")

    # 自由記述3つ
    written1 = request.form.get("written1", "")
    written2 = request.form.get("written2", "")
    written3 = request.form.get("written3", "")

    # タイプごとの素点
    raw = {k: 0 for k in TYPE_KEYS}
    for i in range(num_q):
        val = CHOICE_TO_SCORE.get(answers_raw.get(f"q{i}", "no"), 0)
        if i < len(QUESTION_TYPES):
            tkey = QUESTION_TYPES[i]
        else:
            tkey = TYPE_KEYS[i % len(TYPE_KEYS)]
        raw[tkey] += val

    # ② AI補正用の初期値（0〜+5点を想定）
    bonus = {k: 0 for k in TYPE_KEYS}
    final_scores = raw.copy()

    # ③ Gemini に「ボーナス得点（0〜5の整数）」を考えてもらう
    if gemini_available:
        prompt = f"""
あなたは「RPG風性格診断」の集計AIです。

タイプ一覧:
{json.dumps(TYPE_INFO, ensure_ascii=False, indent=2)}

各タイプの素点(raw_scores)と、ユーザーの自由記述があります。
- written1: 最近よく考えることや悩み
- written2: 日々の行動や習慣
- written3: 理想の自分像

役割:
- raw_scores はベーススコアです。
- 自由記述から見える特徴に応じて、
  各タイプに 0〜5 点のボーナス得点を必要な分だけ加えてください。
- マイナスの補正は行わず、「プラスの補正だけ」を考えてください。

重要な制約:
- bonus_scores の各値は 0〜5 の整数としてください（小数・マイナスは禁止）。
- final_scores[type] = raw_scores[type] + bonus_scores[type] として計算します。

入力:
raw_scores: {json.dumps(raw, ensure_ascii=False)}
written1: {written1}
written2: {written2}
written3: {written3}

出力フォーマット(JSONのみ):
{{
  "bonus_scores": {{"sage": 0, "monk": 0, "priest": 0, "mage": 0, "thief": 0, "artist": 0, "guardian": 0, "commander": 0}}
}}
"""
        data = gemini_generate_json(prompt)
        if data and "bonus_scores" in data:
            try:
                for k in TYPE_KEYS:
                    # 安全側で 0〜5 にクランプ
                    v = int(data["bonus_scores"].get(k, 0))
                    if v < 0:
                        v = 0
                    if v > 5:
                        v = 5
                    bonus[k] = v
                    final_scores[k] = raw[k] + v
            except Exception as e:
                print("Gemini bonus parse error:", e)
                final_scores = raw.copy()
                bonus = {k: 0 for k in TYPE_KEYS}
        else:
            # 失敗時は素点のみ
            final_scores = raw.copy()
            bonus = {k: 0 for k in TYPE_KEYS}
    else:
        # Gemini 未設定ならそのまま素点が最終スコア
        final_scores = raw.copy()
        bonus = {k: 0 for k in TYPE_KEYS}

    # ④ トップタイプ決定（最終スコアベース）
    top_type = max(final_scores, key=lambda k: final_scores[k])

    # ⑤ DB保存
    result_row = DiagnosisResult(
        user_id=user.id,
        top_type=top_type,
        scores=final_scores,
        raw_scores=raw,
        bonus_scores=bonus,
        written1=written1,
        written2=written2,
        written3=written3,
    )
    db.session.add(result_row)
    db.session.commit()

    return redirect(url_for("result"))


# =========================
# タイプ一覧
# =========================
@app.route("/types")
def types():
    return render_template("types.html", type_info=TYPE_INFO, type_keys=TYPE_KEYS)


# =========================
# 快ステータス（画面本体）
# =========================
@app.route("/kai_status")
def kai_status():
    user = get_current_user()
    if not user:
        return redirect(url_for("index"))

    # 画面自体はテンプレ＆JSで描画、ここでは user だけ渡せばOK
    return render_template("kai_status.html", user=user)


# =========================
# 快ログ API
# =========================
@app.route("/api/kai_status")
def api_kai_status():
    """快一覧をJSONで返すAPI"""
    user = ensure_user()
    logs = (
        KaiLog.query.filter_by(user_id=user.id)
        .order_by(KaiLog.id.asc())
        .all()
    )
    data = [{"name": log.name, "count": log.count} for log in logs]
    return jsonify({"ok": True, "logs": data})


@app.route("/register_kai", methods=["POST"])
def register_kai():
    """快を1つ登録 or 実行回数 +1"""
    user = ensure_user()
    data = request.get_json(silent=True) or {}
    name = (data.get("kai") or "").strip()

    if not name:
        return jsonify({"ok": False, "message": "快の名前が空です"}), 400

    log = KaiLog.query.filter_by(user_id=user.id, name=name).first()
    if log:
        log.count = (log.count or 0) + 1
    else:
        log = KaiLog(user_id=user.id, name=name, count=1)
        db.session.add(log)

    db.session.commit()
    return jsonify({"ok": True})


@app.route("/delete_kai", methods=["POST"])
def delete_kai():
    """快を削除"""
    user = ensure_user()
    data = request.get_json(silent=True) or {}
    name = (data.get("kai") or "").strip()

    if not name:
        return jsonify({"ok": False, "message": "快の名前が空です"}), 400

    KaiLog.query.filter_by(user_id=user.id, name=name).delete()
    db.session.commit()
    return jsonify({"ok": True})


# =========================
# Quest: 管理 & 実行
# =========================
@app.route("/admin")
def admin_root():
    """管理トップ。今はクエスト管理に飛ばすだけ。"""
    return redirect(url_for("admin_quests"))


@app.route("/admin/quests")
def admin_quests():
    """クエスト管理用 一覧ページ"""
    quests = Quest.query.order_by(Quest.updated_at.desc()).all()
    return render_template(
        "admin_quests.html",
        quests=quests,
        type_info=TYPE_INFO,
        quest_type_labels=QUEST_TYPE_LABELS,
    )


def _normalize_type_key_from_form(form) -> str:
    """
    フォームから飛んできた type 情報を Quest.type_key 用に正規化。
    - quest_type(複数選択) or type_key 単体 のどちらにも対応
    """
    raw_values = form.getlist("quest_type")
    if not raw_values:
        v = (form.get("type_key") or "").strip()
        if v:
            raw_values = [v]

    chosen = None
    for v in raw_values:
        v = (v or "").strip()
        if not v:
            continue
        chosen = v
        break

    if not chosen:
        return "common"

    return RAW_TYPE_KEY_MAP.get(chosen, "common")


def _parse_steps_json(form):
    """
    quest_create / quest_edit から送られてくる steps_json をパース。
    - 新UI: hidden input の JSON 文字列
    - 旧UI: steps[] の単純なリスト
    """
    steps = []
    steps_str = (form.get("steps_json") or "").strip()
    if steps_str:
        try:
            parsed = json.loads(steps_str)
            steps = parsed
        except Exception as e:
            print("steps_json parse error:", e)
            steps = []

    if not steps:
        # 後方互換用（旧UI）
        steps_raw = form.getlist("steps[]")
        steps = [s.strip() for s in steps_raw if s.strip()]

    return steps


@app.route("/admin/quests/create", methods=["GET", "POST"])
def quest_create():
    """クエスト新規作成"""
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        description = (request.form.get("description") or "").strip()

        type_key = _normalize_type_key_from_form(request.form)
        category = request.form.get("category", "growth")   # UIでは今は出していないが一応保持
        structure = request.form.get("structure", "single")  # 同上

        steps = _parse_steps_json(request.form)

        if not title or not description:
            flash("タイトルと説明は必須です", "quest_error")
            return redirect(url_for("quest_create"))

        quest = Quest(
            title=title,
            description=description,
            type_key=type_key,
            category=category,
            structure=structure,
            steps_json=steps,
        )
        db.session.add(quest)
        db.session.commit()

        flash("クエストを作成しました", "quest_success")
        return redirect(url_for("admin_quests"))

    return render_template(
        "quest_create.html",
        type_info=TYPE_INFO,
        quest_type_labels=QUEST_TYPE_LABELS,
    )


@app.route("/admin/quests/<int:quest_id>/edit", methods=["GET", "POST"])
def admin_quest_edit(quest_id):
    """クエスト編集"""
    quest = Quest.query.get_or_404(quest_id)

    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        description = (request.form.get("description") or "").strip()

        type_key = _normalize_type_key_from_form(request.form)
        category = request.form.get("category", quest.category or "growth")
        structure = request.form.get("structure", quest.structure or "single")

        steps = _parse_steps_json(request.form)

        if not title or not description:
            flash("タイトルと説明は必須です", "quest_error")
            return redirect(url_for("admin_quest_edit", quest_id=quest.id))

        quest.title = title
        quest.description = description
        quest.type_key = type_key
        quest.category = category
        quest.structure = structure
        quest.steps_json = steps

        db.session.commit()
        flash("クエストを更新しました", "quest_success")
        return redirect(url_for("admin_quests"))

    # GET 時は現在値をフォームに表示
    return render_template(
        "quest_edit.html",
        quest=quest,
        type_info=TYPE_INFO,
        quest_type_labels=QUEST_TYPE_LABELS,
    )


@app.route("/admin/quests/<int:quest_id>/delete", methods=["POST"])
def admin_quest_delete(quest_id):
    """クエスト削除"""
    quest = Quest.query.get_or_404(quest_id)

    try:
        # 紐づく進捗も削除
        QuestProgress.query.filter_by(quest_id=quest.id).delete()
        db.session.delete(quest)
        db.session.commit()
        flash("クエストを削除しました", "quest_success")
    except Exception as e:
        db.session.rollback()
        print("admin_quest_delete error:", e)
        flash("クエストの削除中にエラーが発生しました", "quest_error")

    return redirect(url_for("admin_quests"))


@app.route("/admin/quests/success")
def quest_success():
    """
    管理者用：クエスト作成完了のテスト表示ページ。
    実際の運用では flash で十分だけど、
    「quest_success に飛べるようにしたい」用途に対応。
    """
    return render_template("quest_success.html")


# ----- ここから下はプレイヤー向けクエスト入口＆実行 -----
@app.route("/quest")
def quest_top():
    """クエスト入口。今はそのまま一覧に飛ばす。"""
    return redirect(url_for("quest_list"))


@app.route("/quests")
def quest_list():
    """クエスト一覧画面（プレイヤー用）"""
    user = ensure_user()

    # あなたのタイプ（あったら画面に出す用）
    user_type_key = None
    last = (
        DiagnosisResult.query.filter_by(user_id=user.id)
        .order_by(DiagnosisResult.created_at.desc())
        .first()
    )
    if last and last.top_type:
        user_type_key = last.top_type

    quests = Quest.query.order_by(Quest.updated_at.desc()).all()

    # 進捗をまとめて取得
    progresses = QuestProgress.query.filter_by(user_id=user.id).all()
    progress_map = {p.quest_id: p.status for p in progresses}

    # テンプレ用に「dict」のリストに変換
    quests_for_view = []
    for q in quests:
        quests_for_view.append(
            {
                "id": q.id,
                "title": q.title,
                "description": q.description,
                "type": q.type_key or "common",
                "status": progress_map.get(q.id, "not_started"),
            }
        )

    # テンプレが使う type_labels をここで作る
    type_labels = {k: TYPE_INFO[k]["name"] for k in TYPE_KEYS}
    type_labels["common"] = "全タイプ"

    return render_template(
        "quest_list.html",
        quests=quests_for_view,
        type_info=TYPE_INFO,
        quest_type_labels=QUEST_TYPE_LABELS,
        type_labels=type_labels,
        user_type=user_type_key,
        title="クエスト一覧",
    )


@app.route("/quest/<int:quest_id>", methods=["GET", "POST"])
def quest_do(quest_id):
    """
    クエスト実行画面：
    GET  -> quest_do.html（ステップ＋メモ入力フォーム） → 「挑戦中」に更新
    POST -> quest_feedback.html（AIフィードバック表示） → 「クリア済み！」に更新
    """
    user = ensure_user()
    quest = Quest.query.get_or_404(quest_id)

    # そのユーザー＆クエストの進捗レコードを取得／なければ作成
    progress = QuestProgress.query.filter_by(
        user_id=user.id, quest_id=quest.id
    ).first()
    if not progress:
        progress = QuestProgress(
            user_id=user.id,
            quest_id=quest.id,
            status="not_started",
        )
        db.session.add(progress)
        db.session.commit()

    # --- ステップ構造を「共通フォーマット」に正規化する ---
    raw_steps = quest.steps_json or []
    normalized_steps = []

    if isinstance(raw_steps, list):
        for s in raw_steps:
            # ① 文字列だけの旧形式 ["〜を書いてみよう", ...]
            if isinstance(s, str):
                normalized_steps.append(
                    {
                        "title": s,
                        "type": "text",
                        "grid_rows": 0,
                        "grid_cols": 0,
                        "options": [],
                    }
                )
                continue

            # ② dict の場合
            if isinstance(s, dict):
                title = (
                    s.get("title")
                    or s.get("step_title")
                    or s.get("label")
                    or ""
                )

                step_type = (
                    s.get("type")
                    or s.get("step_type")
                    or "text"
                )

                grid_rows = (
                    s.get("grid_rows")
                    or s.get("rows")
                    or s.get("row")
                    or 0
                )
                grid_cols = (
                    s.get("grid_cols")
                    or s.get("cols")
                    or s.get("col")
                    or 0
                )

                options = (
                    s.get("options")
                    or s.get("choices")
                    or s.get("choice")
                    or []
                )
                # 文字列1本で入ってたら改行で区切る
                if isinstance(options, str):
                    options = [
                        o.strip()
                        for o in options.splitlines()
                        if o.strip()
                    ]
                if not isinstance(options, list):
                    options = []

                # 型を揃える
                try:
                    grid_rows = int(grid_rows) if grid_rows else 0
                except Exception:
                    grid_rows = 0
                try:
                    grid_cols = int(grid_cols) if grid_cols else 0
                except Exception:
                    grid_cols = 0

                # ---- ここが今回のポイント ----
                # type が text でも、情報が入っていれば自動で補正する
                if options and step_type == "text":
                    step_type = "choice"
                elif (grid_rows > 0 and grid_cols > 0) and step_type == "text":
                    step_type = "grid"

                # 想定外の値は text に丸める
                if step_type not in ("text", "grid", "choice"):
                    step_type = "text"

                normalized_steps.append(
                    {
                        "title": title,
                        "type": step_type,
                        "grid_rows": grid_rows,
                        "grid_cols": grid_cols,
                        "options": options,
                    }
                )

    # --- POST: クリア扱い ---
    if request.method == "POST":
        raw_feedback = (request.form.get("feedback") or "").strip()

        progress.status = "completed"
        if not progress.started_at:
            progress.started_at = datetime.utcnow()
        progress.completed_at = datetime.utcnow()
        db.session.commit()

        feedback_text = (
            "クエストおつかれさま！\n"
            "今日できたことを、少しだけ自分でほめてあげてみてください。"
        )

        if gemini_available and raw_feedback:
            prompt = f"""
あなたは「クエストの振り返りコーチ」です。
以下は、ユーザーがクエストに取り組んで感じたこと・気づいたことのメモです。

{raw_feedback}

この人が「やってよかったな」と少し安心できるような、
あたたかいフィードバックを日本語で150〜250文字で書いてください。

・説教しない
・反省点を掘り返しすぎない
・できたことを1〜2個だけ見つけてあげる
"""
            ai_fb = gemini_generate_text(prompt)
            if ai_fb:
                feedback_text = ai_fb

        return render_template(
            "quest_feedback.html",
            quest=quest,
            raw_feedback=raw_feedback,
            feedback=feedback_text,
        )

    # --- GET: 画面を開いたら「挑戦中」にする ---
    if progress.status == "not_started":
        progress.status = "in_progress"
        progress.started_at = datetime.utcnow()
        db.session.commit()

    # 正規化済みステップをテンプレに渡す
    return render_template(
        "quest_do.html",
        quest=quest,
        steps=normalized_steps,
        type_info=TYPE_INFO,
    )


# =========================
# Journal（日記）
# =========================
@app.route("/journal", methods=["GET", "POST"])
def journal():
    user = ensure_user()
    if request.method == "POST":
        content = (request.form.get("content") or "").strip()
        if not content:
            flash("日記の内容を入力してね")
            return redirect(url_for("journal"))
        entry = JournalEntry(user_id=user.id, content=content)
        db.session.add(entry)
        db.session.commit()
        flash("日記を保存しました")
        return redirect(url_for("journal"))

    # プレビューから戻ってきたときだけドラフトを拾う
    draft = None
    if request.args.get("mode") == "edit":
        draft = session.pop("journal_draft", None)

    entries = (
        JournalEntry.query.filter_by(user_id=user.id)
        .order_by(JournalEntry.created_at.desc())
        .all()
    )
    return render_template("journal.html", entries=entries, user=user, draft=draft)


@app.route("/journal/compose", methods=["POST"])
def journal_compose():
    """6ステップ日記を AI でまとめてプレビュー画面に飛ばす"""

    user = ensure_user()

    # 6ステップの内容をまとめて1つのテキストにする
    steps = []
    for i in range(1, 7):
        val = (request.form.get(f"step{i}") or "").strip()
        if val:
            steps.append(f"【ステップ{i}】{val}")

    base_text = "\n".join(steps).strip()
    if not base_text:
        base_text = (request.form.get("content") or "").strip()

    if not base_text:
        flash("日記の内容を入力してね")
        return redirect(url_for("journal"))

    # Gemini に「やさしく整形して」とお願い
    prompt = f"""
あなたは「心にやさしい編集者」です。
以下の文章を、書き手の感情を大切にしながら、読みやすい日記文に整えてください。

- 批判や否定はしない
- 上から目線にならない
- できればポジティブな視点を 1つだけ添える

元のテキスト:
{base_text}
"""
    composed = gemini_generate_text(prompt) or base_text

    # 通常日記モード用のドラフトとして一時保存
    session["journal_draft"] = composed

    # プレビュー画面へ
    return render_template(
        "journal_preview.html",
        composed=composed,
        analysis=None,
        user=user,
    )


@app.route("/journal/feedback", methods=["POST"])
def journal_feedback():
    user = ensure_user()

    # どの日記か特定
    entry_id = request.form.get("entry_id")
    if not entry_id:
        flash("対象の日記が見つかりませんでした")
        return redirect(url_for("journal"))

    entry = JournalEntry.query.filter_by(id=entry_id, user_id=user.id).first()
    if not entry:
        flash("対象の日記が見つかりませんでした")
        return redirect(url_for("journal"))

    text = (entry.content or "").strip()
    if not text:
        flash("日記の内容が空でした")
        return redirect(url_for("journal"))

    # AI にやさしい感想を書いてもらう
    prompt = f"""
あなたは日記にそっと寄り添うカウンセラーです。
以下の日記を読んで、書き手が少しラクになれるようなコメントを
やさしい日本語で 120〜200文字くらいで書いてください。

- 説教しない
- ダメ出ししない
- できていることを1〜2個だけ拾ってあげる

日記:
{text}
"""
    ai_comment = gemini_generate_text(prompt)

    if ai_comment:
        entry.feedback = ai_comment
        db.session.commit()
    else:
        flash("AIの感想生成に失敗しました")

    # 再びジャーナル画面へ
    return redirect(url_for("journal"))


@app.route("/journal/save", methods=["POST"])
def journal_save():
    user = ensure_user()

    # ① 日記本文
    content = (request.form.get("content") or "").strip()
    if not content:
        flash("日記の内容を入力してね")
        return redirect(url_for("journal"))

    # まずは日記だけを確実に保存する
    entry = JournalEntry(user_id=user.id, content=content)
    db.session.add(entry)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("journal_save entry commit error:", e)
        flash("日記の保存中にエラーが起きました。時間をおいてもう一度試してね。")
        return redirect(url_for("journal"))

    # ② 抽出された「快」を受け取る（JSON文字列）
    kai_json = request.form.get("kai_extracted") or ""
    kai_list: list[str] = []
    if kai_json:
        try:
            parsed = json.loads(kai_json)
            if isinstance(parsed, list):
                kai_list = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception as e:
            print("kai_extracted JSON parse error:", e)
            kai_list = []

    # ③ 快を KaiLog に反映（あれば count+1、なければ新規作成）
    if kai_list:
        try:
            # すでに登録されている快（このユーザー分）をまとめて取得
            existing_logs = KaiLog.query.filter_by(user_id=user.id).all()

            for raw_name in kai_list:
                name = (str(raw_name) or "").strip()
                if not name:
                    continue

                # 似ている「快」がすでにあるかどうかチェック
                similar_log = find_similar_kai(existing_logs, name, threshold=0.7)

                if similar_log:
                    # 既存の快としてカウントだけ増やす
                    similar_log.count = (similar_log.count or 0) + 1
                else:
                    # 新しい快として登録
                    log = KaiLog(user_id=user.id, name=name, count=1)
                    db.session.add(log)
                    # 次のループからも類似判定に使えるようにリストにも追加
                    existing_logs.append(log)

            db.session.commit()
        except Exception as e:
            # ここでエラーが出ても日記自体はすでに保存済みなので、
            # ロールバックしてログだけ出しておく
            db.session.rollback()
            print("KaiLog update error:", e)

    flash("日記を保存しました")
    return redirect(url_for("journal"))


@app.route("/journal/extract_kai", methods=["POST"])
def journal_extract_kai():
    data = request.get_json(silent=True) or {}
    text = (data.get("content") or "").strip()

    if not text:
        return jsonify({"ok": False, "kai": []})

    if not gemini_available:
        return jsonify({"ok": False, "kai": []})

    prompt = f"""
あなたは「快（心地よさ）」発見の専門家です。
以下の日記文の中から、「その人にとっての快（心地よさ・好きなこと・大切にしたいこと）」を
3〜5個、短い一文で抽出してください。

- 箇条書きで
- やさしい日本語で
- 出力は箇条書きだけにしてください

日記文:
{text}
"""
    result = gemini_generate_text(prompt) or ""
    kai_list = []

    for line in result.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("・-＊*•●■□0123456789.①②③④⑤ ").strip()
        if line:
            kai_list.append(line)

    return jsonify({"ok": True, "kai": kai_list})


@app.route("/journal/delete/<int:entry_id>", methods=["POST"])
def journal_delete(entry_id):
    user = ensure_user()

    # 自分の日記かチェック
    entry = JournalEntry.query.filter_by(id=entry_id, user_id=user.id).first()
    if not entry:
        flash("対象の日記が見つかりませんでした")
        return redirect(url_for("journal"))

    try:
        db.session.delete(entry)
        db.session.commit()
        flash("日記を削除しました")
    except Exception as e:
        db.session.rollback()
        print("journal_delete error:", e)
        flash("日記の削除中にエラーが発生しました。時間をおいてもう一度試してね。")

    return redirect(url_for("journal"))


# =========================
# クレジット
# =========================
@app.route("/credit")
def credit():
    user = get_current_user()
    return render_template("credit.html", user=user)


# =========================
# Reset & init
# =========================
@app.route("/reset")
def reset():
    user = get_current_user()
    if user:
        DiagnosisResult.query.filter_by(user_id=user.id).delete()
        JournalEntry.query.filter_by(user_id=user.id).delete()
        KaiLog.query.filter_by(user_id=user.id).delete()
        QuestProgress.query.filter_by(user_id=user.id).delete()
        db.session.commit()
    session.clear()
    return redirect(url_for("index"))


@app.cli.command("init-db")
def init_db():
    """flask init-db"""
    with app.app_context():
        db.create_all()
    print("DB initialized")


if __name__ == "__main__":
    print("Gemini available:", gemini_available)
    print("GEMINI_API_KEY head:", repr(GEMINI_API_KEY[:8]))
    print("GEMINI_MODEL_TEXT:", GEMINI_MODEL_TEXT)
    print("DB URI:", app.config["SQLALCHEMY_DATABASE_URI"])
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
