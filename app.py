import os
import json
import secrets
import logging
from functools import wraps
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo  # JST用
from difflib import SequenceMatcher
from flask import Flask, render_template, request, redirect, url_for, jsonify

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
    abort,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import text

# =========================
# ベースパス & .env
# =========================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# =========================
# Production logging
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mindquest")

# =========================
# Gemini 設定（Google Gen AI SDK / google-genai）
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_TEXT = os.getenv("GEMINI_MODEL_TEXT", "gemini-2.5-flash-lite")
GEMINI_MODEL_JSON = os.getenv("GEMINI_MODEL_JSON", "gemini-2.5-flash-lite")

gemini_available = False
gemini_client = None

try:
    from google import genai
    from google.genai import types as genai_types

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_available = True
except Exception as e:
    logger.exception("Gemini initialization failed.")
    gemini_available = False
    gemini_client = None


def gemini_generate_text(
    prompt,
    model_name=None,
    temperature=0.6,
    max_tokens=1024,
):
    """Geminiテキスト生成ヘルパー。本番ログにプロンプト/APIキー/本文を出さない。"""
    if not gemini_available or gemini_client is None:
        logger.warning("Gemini is unavailable. Text generation skipped.")
        return None

    model_name = model_name or GEMINI_MODEL_TEXT

    try:
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        try:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=0,
                include_thoughts=False,
            )
        except Exception:
            # SDK/モデルによって非対応の場合があるため、落とさず続行する
            pass

        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )

        return (resp.text or "").strip()

    except Exception:
        logger.exception("Gemini text generation failed.")
        return None


def gemini_generate_json(
    prompt,
    model_name=None,
    temperature=0.2,
    max_tokens=768,
):
    """JSONを返したいとき用。response_mime_type でJSON返却を安定化。"""
    if not gemini_available or gemini_client is None:
        logger.warning("Gemini is unavailable. JSON generation skipped.")
        return None

    model_name = model_name or GEMINI_MODEL_JSON

    try:
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )

        txt = (resp.text or "").strip()

        import json as _json
        return _json.loads(txt)

    except Exception:
        logger.exception("Gemini JSON generation failed.")
        return None



# =========================
# Kai AI Analysis
# =========================
def analyze_kai(kai_list):
    """
    保存済みの快ログをもとに、シンプルなAI分析を返す。
    MAX_TOKENS対策として、AIに渡す快ログは最大8件に絞り、
    出力も300〜450文字程度の短めにする。
    """
    if not kai_list:
        return "まだ快のデータがありません。まずは1つ、心地よかったことを登録してみてください。"

    rows = []
    for log in kai_list[:8]:
        name = (log.name or "").strip()
        if not name:
            continue
        count = log.count or 0
        rows.append(f"- {name}（{count}回）")

    if not rows:
        return "まだ分析できる快のデータがありません。"

    text_data = "\n".join(rows)

    prompt = f"""
最近の快ログ：
{text_data}

以下の形式だけで、合計300〜450文字で完結させてください。
あいさつ文は不要。医療・診断表現は避け、断定しすぎない。
必ず3つ目の行動まで出してください。

🧠 あなたの快傾向
2文以内。

⚠️ 最近不足しているかもしれない快
1文以内。

🎯 今日のおすすめ行動
1. 5分でできる行動
2. すぐできる気分転換
3. 夜にできる振り返り
"""

    result = gemini_generate_text(
        prompt,
        temperature=0.2,
        max_tokens=4096,
    )

    if not result:
        return "AI分析の生成に失敗しました。GEMINI_API_KEYやモデル設定を確認してください。"

    return result


# =========================
# Flask & DB 設定（Neon 前提）
# =========================
class Config:
    """本番運用向けの基本設定。

    Render/本番では以下を環境変数で必ず設定:
    - DATABASE_URL
    - SECRET_KEY
    - GEMINI_API_KEY（AI機能を使う場合）
    """
    ENV_NAME = os.getenv("FLASK_ENV", os.getenv("APP_ENV", "production")).lower()
    IS_PRODUCTION = ENV_NAME in {"production", "prod"}

    SECRET_KEY = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        if IS_PRODUCTION:
            raise RuntimeError("SECRET_KEY が設定されていません。本番では必ず環境変数に設定してください。")
        SECRET_KEY = "dev-secret-key"

    SESSION_COOKIE_SECURE = os.getenv(
        "SESSION_COOKIE_SECURE",
        "True" if IS_PRODUCTION else "False",
    ) == "True"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)

    raw_url = os.getenv("DATABASE_URL", "")
    if not raw_url:
        raise RuntimeError("DATABASE_URL が設定されていません。Neon のURLを環境変数に設定してください。")

    SQLALCHEMY_DATABASE_URI = raw_url.replace("postgres://", "postgresql://", 1)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 280,
    }


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# =========================
# Admin auth 設定
# =========================
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


def admin_required(func):
    """管理画面専用ガード。既存ユーザー機能とは分離して管理者だけ通す。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            next_url = request.full_path if request.query_string else request.path
            return redirect(url_for("admin_login", next=next_url))
        return func(*args, **kwargs)

    return wrapper

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
    xp = db.Column(db.Integer, default=0)
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
    # ★ 表示順（小さいほど上に表示 / 管理画面で編集）
    sort_order = db.Column(db.Integer, nullable=False, default=0)


class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    feedback = db.Column(db.Text)  # AIの感想を保存する
    mood_score = db.Column(db.Integer)  # 気分スコア（1〜5 / 任意）


class KaiLog(db.Model):
    __tablename__ = "kai_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(128), nullable=False)  # 快の名前
    count = db.Column(db.Integer, default=0)          # 実行回数
    # 最初の作成時 or 直近追加時刻（register_kai 側で更新する）
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class KaiAnalysisLog(db.Model):
    """
    快分析の履歴。
    AI分析を実行するたびに、その時点の分析結果をユーザー別に保存する。
    """
    __tablename__ = "kai_analysis_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    analysis_text = db.Column(db.Text, nullable=False)
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


class QuestLog(db.Model):
    __tablename__ = "quest_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    quest_id = db.Column(db.Integer, db.ForeignKey("quest.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # ユーザーが書いた振り返りメモ（最後のコメント欄）
    raw_feedback = db.Column(db.Text)

    # AIからのフィードバック（実際に画面に出した文章）
    ai_feedback = db.Column(db.Text)

    # クエストのステップと、各ステップに対するユーザーの回答
    steps_data = db.Column(db.JSON)


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

MOOD_LABELS = {
    1: "かなりしんどい",
    2: "少ししんどい",
    3: "ふつう",
    4: "まあまあ良い",
    5: "とても良い",
}


def parse_mood_score(value):
    """フォームやAPIから来た気分スコアを 1〜5 の整数に丸める。未入力は None。"""
    if value in (None, ""):
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score

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

# 質問→タイプの割り当て（元からあった想定のやつをここで定義）
QUESTION_TYPES = [
    "sage", "sage", "sage", "sage",
    "monk", "monk", "monk", "monk",
    "priest", "priest", "priest", "priest",
    "artist", "artist", "artist", "artist",
    "thief", "thief", "thief", "thief",
    "mage", "mage", "mage", "mage",
    "guardian", "guardian", "guardian", "guardian",
    "commander", "commander", "commander", "commander",
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


MAX_LEVEL = 99
BASE_REQUIRED_XP = 20
XP_GROWTH_RATE = 1.12


def required_xp_for_next(level: int) -> int:
    """
    そのレベルから次のレベルに上がるために必要なXP。
    Lv1→2を20XPとして、以降は指数関数的に増える。
    Lv99は上限なので次レベルなし。
    """
    level = max(1, int(level or 1))

    if level >= MAX_LEVEL:
        return 0

    return max(1, round(BASE_REQUIRED_XP * (XP_GROWTH_RATE ** (level - 1))))


def calc_level(xp: int) -> int:
    """
    累計XPから現在レベルを計算する。
    最高レベルは99。
    """
    xp = xp or 0
    level = 1
    remaining_xp = xp

    while level < MAX_LEVEL:
        need = required_xp_for_next(level)

        if remaining_xp < need:
            return level

        remaining_xp -= need
        level += 1

    return MAX_LEVEL


def get_level_progress(xp: int):
    """
    XPバー表示用の進捗情報。
    Lv99の場合は進捗100%固定。
    """
    xp = xp or 0
    level = calc_level(xp)

    if level >= MAX_LEVEL:
        return {
            "level": MAX_LEVEL,
            "xp": xp,
            "current_level_xp": xp,
            "next_level_xp": xp,
            "xp_in_current_level": 0,
            "required_for_next": 0,
            "remaining_xp": 0,
            "progress": 100,
            "is_max_level": True,
        }

    remaining_xp = xp
    total_before_level = 0

    for lv in range(1, level):
        need = required_xp_for_next(lv)
        remaining_xp -= need
        total_before_level += need

    need = required_xp_for_next(level)
    progress = int((remaining_xp / need) * 100) if need > 0 else 100

    return {
        "level": level,
        "xp": xp,
        "current_level_xp": total_before_level,
        "next_level_xp": total_before_level + need,
        "xp_in_current_level": remaining_xp,
        "required_for_next": need,
        "remaining_xp": need - remaining_xp,
        "progress": max(0, min(100, progress)),
        "is_max_level": False,
    }


def add_xp(user, amount: int, reason: str = ""):
    """
    ユーザーにXPを加算して、レベルも更新する。
    db.session.commit() は呼び出し元の処理に任せる。

    return:
      True  = レベルアップした
      False = レベルアップしていない
    """
    if not user or amount <= 0:
        return False

    old_level = user.level or 1
    old_xp = user.xp or 0

    user.xp = old_xp + amount
    user.level = calc_level(user.xp)

    leveled_up = user.level > old_level

    logger.info(
        "XP added amount=%s reason=%s old_xp=%s total_xp=%s old_level=%s level=%s",
        amount,
        reason,
        old_xp,
        user.xp,
        old_level,
        user.level,
    )

    return leveled_up


def ensure_extra_columns():
    """
    既存DB向けの軽量マイグレーション。
    Neon/PostgreSQL上の既存 user テーブルに xp カラムが無い場合だけ追加する。
    """
    try:
        db.session.execute(text('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS xp INTEGER DEFAULT 0;'))
        db.session.execute(text('UPDATE "user" SET xp = 0 WHERE xp IS NULL;'))
        db.session.execute(text('UPDATE "user" SET level = 1 WHERE level IS NULL;'))

        # 気持ちグラフ用：既存Neon DBにも安全に追加
        db.session.execute(text('ALTER TABLE journal_entry ADD COLUMN IF NOT EXISTS mood_score INTEGER;'))

        db.session.commit()
        logger.info("Extra columns checked: xp / mood_score OK")
    except Exception as e:
        db.session.rollback()
        logger.exception("ensure_extra_columns failed.")


@app.context_processor
def inject_global_status():
    """
    base.html で共通表示するためのグローバル情報。
    - user
    - xp_progress
    を全テンプレートで使えるようにする。
    """
    try:
        user = get_current_user()
        if not user:
            return {}

        return {
            "user": user,
            "xp_progress": get_level_progress(user.xp or 0),
        }
    except Exception as e:
        logger.exception("inject_global_status failed.")
        return {}


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

        # もし他のユーザーがすでにこのメールを使っていたらエラー
        if email:
            existing = User.query.filter(
                User.email == email,
                User.id != user.id,
            ).first()
            if existing:
                flash(
                    "このメールアドレスはすでに登録されています。ログイン画面からログインしてください。",
                    "account_error",
                )
                return redirect(url_for("login"))

            # メール更新
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
    KaiAnalysisLog.query.filter_by(user_id=user.id).delete()
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

        # ユーザーが存在しない or パスワードが未設定 or パスワード違う
        if (not user) or (not user.password_hash) or (not check_password_hash(user.password_hash, password)):
            return render_template(
                "login.html",
                error="メールアドレスまたはパスワードが違います。",
            )

        # 🔑 ログイン成功したら、このブラウザの user_token を
        #    「ログインしたユーザーの user_token」に差し替える
        resp = make_response(redirect(url_for("menu")))

        resp.set_cookie(
            "user_token",
            user.user_token,
            httponly=True,
            secure=app.config["SESSION_COOKIE_SECURE"],
            samesite="Lax",
            max_age=60 * 60 * 24 * 365,  # 1年
        )

        # session["user_id"] は使っていないのでクリアでOK
        session.clear()

        return resp

    return render_template("login.html")


@app.route("/logout")
def logout():
    # ログアウトしたらトップに戻す
    resp = make_response(redirect(url_for("index")))
    # この端末用の user_token クッキーを削除
    resp.delete_cookie("user_token")
    # セッションもきれいにしておく
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

    # メニュー画面用：快ランキングTOP5
    top_kai = (
        KaiLog.query
        .filter_by(user_id=user.id)
        .order_by(KaiLog.count.desc(), KaiLog.created_at.desc())
        .limit(5)
        .all()
    )

    return render_template(
        "menu.html",
        user=user,
        name=display_name,
        user_type=user_type,
        level=level,
        top_kai=top_kai,
    )


# =========================
# 冒険ログ（クエスト + 診断）
# =========================
@app.route("/logs")
def logs():
    user = ensure_user()

    # クエストログ
    quest_logs = (
        db.session.query(QuestLog, Quest)
        .join(Quest, QuestLog.quest_id == Quest.id)
        .filter(QuestLog.user_id == user.id)
        .order_by(QuestLog.created_at.desc())
        .limit(50)
        .all()
    )

    # 診断ログ
    diagnosis_logs = (
        DiagnosisResult.query.filter_by(user_id=user.id)
        .order_by(DiagnosisResult.created_at.desc())
        .limit(50)
        .all()
    )

    return render_template(
        "logs.html",
        user=user,
        quest_logs=quest_logs,
        diagnosis_logs=diagnosis_logs,
        type_info=TYPE_INFO,
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
                logger.exception("Gemini bonus parse failed.")
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

    analysis_logs = (
        KaiAnalysisLog.query
        .filter_by(user_id=user.id)
        .order_by(KaiAnalysisLog.created_at.desc())
        .limit(5)
        .all()
    )

    # 画面自体はテンプレ＆JSで描画、ここでは user と analysis 初期値と履歴を渡す
    return render_template(
        "kai_status.html",
        user=user,
        analysis=None,
        analysis_logs=analysis_logs,
    )


@app.route("/kai_analyze")
def kai_analyze():
    """快ログをAI分析して、分析結果を保存し、kai_status.html に表示する。"""
    user = ensure_user()

    kai_list = (
        KaiLog.query
        .filter_by(user_id=user.id)
        .order_by(KaiLog.created_at.desc())
        .limit(50)
        .all()
    )

    analysis = analyze_kai(kai_list)

    # 分析結果を履歴として保存する。
    # 「まだデータがない」系の案内文も残しておくと、初回体験の流れを後から確認できる。
    try:
        if analysis:
            log = KaiAnalysisLog(
                user_id=user.id,
                analysis_text=analysis,
            )
            db.session.add(log)
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("KaiAnalysisLog save failed.")

    analysis_logs = (
        KaiAnalysisLog.query
        .filter_by(user_id=user.id)
        .order_by(KaiAnalysisLog.created_at.desc())
        .limit(5)
        .all()
    )

    return render_template(
        "kai_status.html",
        user=user,
        analysis=analysis,
        analysis_logs=analysis_logs,
    )


@app.route("/kai_analysis/<int:log_id>/delete", methods=["POST"])
def kai_analysis_delete(log_id):
    """快分析履歴を1件削除する。自分の履歴だけ削除可能。"""
    user = ensure_user()

    log = KaiAnalysisLog.query.filter_by(
        id=log_id,
        user_id=user.id,
    ).first()

    if not log:
        flash("削除対象の分析履歴が見つかりませんでした。")
        return redirect(url_for("kai_status"))

    try:
        db.session.delete(log)
        db.session.commit()
        flash("快分析履歴を削除しました。")
    except Exception as e:
        db.session.rollback()
        logger.exception("kai_analysis_delete failed.")
        flash("快分析履歴の削除中にエラーが発生しました。")

    return redirect(url_for("kai_status"))


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

    data = []
    for log in logs:
        data.append(
            {
                "id": log.id,
                "name": log.name,
                "count": log.count or 0,
                # JS 側で扱いやすいように ISO 文字列で渡す
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
        )

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
        # すでにあるなら回数+1 と同時に「最終追加日」を更新
        log.count = (log.count or 0) + 1
        log.created_at = datetime.utcnow()
        leveled_up = add_xp(user, 3, f"快カウント: {log.name}")
    else:
        # 新規なら1回目として作成
        log = KaiLog(user_id=user.id, name=name, count=1, created_at=datetime.utcnow())
        db.session.add(log)
        leveled_up = add_xp(user, 3, f"快登録: {name}")

    db.session.commit()
    return jsonify({
        "ok": True,
        "xp": user.xp or 0,
        "level": user.level or 1,
        "leveled_up": leveled_up,
    })


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
# Admin Login
# =========================
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """管理者ログイン。ADMIN_PASSWORD は .env / Render 環境変数で設定する。"""
    if session.get("is_admin"):
        return redirect(url_for("admin_quests"))

    next_url = request.args.get("next") or url_for("admin_quests")

    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        next_url = request.form.get("next") or url_for("admin_quests")

        if not ADMIN_PASSWORD:
            logger.error("ADMIN_PASSWORD is not set.")
            flash("管理者パスワードが設定されていません。環境変数 ADMIN_PASSWORD を設定してください。", "error")
            return render_template("admin_login.html", next_url=next_url)

        if password == ADMIN_PASSWORD:
            session.clear()
            session["is_admin"] = True
            session.permanent = True
            flash("管理者としてログインしました。", "success")
            return redirect(next_url if next_url.startswith("/") else url_for("admin_quests"))

        flash("管理者パスワードが違います。", "error")

    return render_template("admin_login.html", next_url=next_url)


@app.route("/admin/logout")
def admin_logout():
    """管理者ログアウト。通常ユーザーの user_token Cookie は消さない。"""
    session.pop("is_admin", None)
    flash("管理者ログアウトしました。", "success")
    return redirect(url_for("admin_login"))


# =========================
# Quest: 管理 & 実行
# =========================
@app.route("/admin")
@admin_required
def admin_root():
    """管理トップ。今はクエスト管理に飛ばすだけ。"""
    return redirect(url_for("admin_quests"))


@app.route("/admin/quests")
@admin_required
def admin_quests():
    """クエスト管理用 一覧ページ"""
    quests = (
        Quest.query
        .order_by(Quest.sort_order.asc(), Quest.updated_at.desc())
        .all()
    )
    return render_template(
        "admin_quests.html",
        quests=quests,
        type_info=TYPE_INFO,
        quest_type_labels=QUEST_TYPE_LABELS,
    )


@app.route("/admin/quests/reorder", methods=["POST"])
@admin_required
def admin_quests_reorder():
    """表示順(sort_order)を一括更新"""
    quests = Quest.query.all()
    for q in quests:
        key = f"order_{q.id}"
        if key in request.form:
            try:
                q.sort_order = int(request.form[key])
            except ValueError:
                # 数値じゃない場合はスキップ
                continue
    db.session.commit()
    flash("クエストの並び順を保存しました。", "quest_success")
    return redirect(url_for("admin_quests"))


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
            logger.exception("steps_json parse failed.")
            steps = []

    if not steps:
        # 後方互換用（旧UI）
        steps_raw = form.getlist("steps[]")
        steps = [s.strip() for s in steps_raw if s.strip()]

    return steps


@app.route("/admin/quests/create", methods=["GET", "POST"])
@admin_required
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
@admin_required
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
@admin_required
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
        logger.exception("admin_quest_delete failed.")
        flash("クエストの削除中にエラーが発生しました", "quest_error")

    return redirect(url_for("admin_quests"))


@app.route("/admin/quests/success")
@admin_required
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

    quests = (
        Quest.query
        .order_by(Quest.sort_order.asc(), Quest.updated_at.desc())
        .all()
    )

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

        # ★ 各ステップの回答をフォームから回収
        steps_data = []
        for idx, step in enumerate(normalized_steps):
            answer = (request.form.get(f"step_{idx}") or "").strip()
            steps_data.append(
                {
                    "title": step["title"],
                    "type": step["type"],
                    "grid_rows": step["grid_rows"],
                    "grid_cols": step["grid_cols"],
                    "options": step["options"],
                    "answer": answer,
                }
            )

        # 進捗を完了に更新
        progress.status = "completed"
        if not progress.started_at:
            progress.started_at = datetime.utcnow()
        progress.completed_at = datetime.utcnow()
        leveled_up = add_xp(user, 30, "クエストクリア")

        # デフォルトのフィードバック文
        feedback_text = (
            "クエストおつかれさま！\n"
            "今日できたことを、少しだけ自分でほめてあげてみてください。"
        )

        # ★ ステップの回答 + 最後のひとことメモ をまとめて AI に渡す
        if gemini_available:
            # ステップ回答をテキストに整形
            step_lines = []
            for idx, s in enumerate(steps_data):
                ans = (s.get("answer") or "").strip()
                if not ans:
                    continue
                title = s.get("title") or f"STEP {idx+1}"
                step_lines.append(f"STEP {idx+1}：{title}\n{ans}")
            all_step_text = "\n\n".join(step_lines).strip()

            # 何かしら入力があるときだけ呼ぶ
            if all_step_text or raw_feedback:
                prompt = f"""
あなたは「クエストの振り返りコーチ」です。

以下は、ユーザーがクエストに取り組んだときのメモです。

[クエストの各ステップと回答]
{all_step_text or "（回答はありません）"}

[最後のひとことメモ]
{raw_feedback or "（入力なし）"}

これらすべてを踏まえて、
・良かった点
・気づいたこと
・次に踏み出せそうな一歩
を、やさしい日本語で約200文字でまとめてください。
説教やダメ出しはせず、できている点を1〜2個だけそっと伝えてください。
"""
                ai_fb = gemini_generate_text(prompt)
                if ai_fb:
                    feedback_text = ai_fb

        # クエストログを保存（ステップ内容も含めて）
        log = QuestLog(
            user_id=user.id,
            quest_id=quest.id,
            raw_feedback=raw_feedback,
            ai_feedback=feedback_text,
            steps_data=steps_data,
        )
        db.session.add(log)
        db.session.commit()

        if leveled_up:
            flash(f"🎉 レベルアップ！ Lv{user.level}になった！", "levelup")
        else:
            flash("クエストクリア！ +30XP！", "xp")

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
    draft_mood_score = None
    if request.args.get("mode") == "edit":
        draft = session.pop("journal_draft", None)
        draft_mood_score = session.pop("journal_draft_mood_score", None)

    entries = (
        JournalEntry.query.filter_by(user_id=user.id)
        .order_by(JournalEntry.created_at.desc())
        .all()
    )
    return render_template(
        "journal.html",
        entries=entries,
        user=user,
        draft=draft,
        draft_mood_score=draft_mood_score,
        mood_labels=MOOD_LABELS,
    )


@app.route("/journal/compose", methods=["POST"])
def journal_compose():
    """6ステップ日記を AI でまとめてプレビュー画面に飛ばす"""

    user = ensure_user()
    mood_score = parse_mood_score(request.form.get("mood_score"))

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
    session["journal_draft_mood_score"] = mood_score

    # プレビュー画面へ
    return render_template(
        "journal_preview.html",
        composed=composed,
        analysis=None,
        user=user,
        mood_score=mood_score,
        mood_label=MOOD_LABELS.get(mood_score),
        mood_labels=MOOD_LABELS,
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
        leveled_up = add_xp(user, 5, "AI感想")
        db.session.commit()

        if leveled_up:
            flash(f"🎉 レベルアップ！ Lv{user.level}になった！", "levelup")
        else:
            flash("AIの感想を保存しました。+5XP！", "xp")
    else:
        flash("AIの感想生成に失敗しました", "error")

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

    # 気分スコア（任意）
    mood_score = parse_mood_score(request.form.get("mood_score"))

    # まずは日記だけを確実に保存する
    entry = JournalEntry(user_id=user.id, content=content, mood_score=mood_score)
    db.session.add(entry)
    journal_leveled_up = False

    try:
        journal_leveled_up = add_xp(user, 10, "日記保存")
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("journal_save entry commit failed.")
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
            logger.exception("kai_extracted JSON parse failed.")
            kai_list = []

    # ③ 快を KaiLog に反映（あれば count+1、なければ新規作成）
    kai_leveled_up = False

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
                    if add_xp(user, 3, f"快カウント: {similar_log.name}"):
                        kai_leveled_up = True
                else:
                    # 新しい快として登録
                    log = KaiLog(user_id=user.id, name=name, count=1)
                    db.session.add(log)
                    # 次のループからも類似判定に使えるようにリストにも追加
                    existing_logs.append(log)
                    if add_xp(user, 3, f"快登録: {name}"):
                        kai_leveled_up = True

            db.session.commit()
        except Exception as e:
            # ここでエラーが出ても日記自体はすでに保存済みなので、
            # ロールバックしてログだけ出しておく
            db.session.rollback()
            logger.exception("KaiLog update failed.")

    if journal_leveled_up or kai_leveled_up:
        flash(f"🎉 レベルアップ！ Lv{user.level}になった！", "levelup")
    elif kai_list:
        flash(f"日記を保存しました。+10XP！ 快も{len(kai_list)}件登録！", "xp")
    else:
        flash("日記を保存しました。+10XP！", "xp")

    return redirect(url_for("journal"))


@app.route("/api/mood_graph")
def api_mood_graph():
    """
    気持ちグラフ用API。
    ?range=30 / 90 / 180 / 365 に対応。
    同じ日に複数の日記がある場合は、その日の平均気分を返す。
    """
    user = ensure_user()

    try:
        range_days = int(request.args.get("range", 30))
    except (TypeError, ValueError):
        range_days = 30

    allowed_ranges = {30, 90, 180, 365}
    if range_days not in allowed_ranges:
        range_days = 30

    # DBはUTC想定。JST表示とのズレを避けるため少し余裕を持って取得する。
    since_utc = datetime.utcnow() - timedelta(days=range_days + 2)

    entries = (
        JournalEntry.query
        .filter(
            JournalEntry.user_id == user.id,
            JournalEntry.mood_score.isnot(None),
            JournalEntry.created_at >= since_utc,
        )
        .order_by(JournalEntry.created_at.asc())
        .all()
    )

    today_jst = datetime.now(JST).date()
    start_date_jst = today_jst - timedelta(days=range_days - 1)

    grouped = {}
    for entry in entries:
        created_at = entry.created_at
        if created_at is None:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        created_jst = created_at.astimezone(JST)
        date_obj = created_jst.date()

        if date_obj < start_date_jst or date_obj > today_jst:
            continue

        date_key = date_obj.isoformat()
        grouped.setdefault(date_key, [])
        grouped[date_key].append(
            {
                "id": entry.id,
                "time": created_jst.strftime("%H:%M"),
                "mood_score": entry.mood_score,
                "mood_label": MOOD_LABELS.get(entry.mood_score, "未記録"),
                "content": entry.content or "",
            }
        )

    points = []
    for date_key in sorted(grouped.keys()):
        day_entries = grouped[date_key]
        scores = [e["mood_score"] for e in day_entries if e.get("mood_score") is not None]
        if not scores:
            continue

        # 同じ日に複数の日記がある場合は、その日の平均をグラフの1点にする
        avg_score = round(sum(scores) / len(scores), 2)

        points.append(
            {
                "date": date_key,
                "avg_mood": avg_score,
                "avg_label": MOOD_LABELS.get(round(avg_score), ""),
                "count": len(day_entries),
                "entries": day_entries,
            }
        )

    # 期間中の平均気分を計算
    # グラフの1点＝1日なので、日ごとの平均をさらに平均する。
    # これで「記録した日」を均等に扱える。
    daily_scores = [p["avg_mood"] for p in points if p.get("avg_mood") is not None]
    average = round(sum(daily_scores) / len(daily_scores), 1) if daily_scores else None
    record_days = len(daily_scores)
    record_count = sum(p.get("count", 0) for p in points)

    return jsonify(
        {
            "ok": True,
            "range": range_days,
            "points": points,
            "average": average,
            "average_label": MOOD_LABELS.get(round(average), "") if average is not None else None,
            "record_days": record_days,
            "record_count": record_count,
        }
    )


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
        logger.exception("journal_delete failed.")
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
@app.route("/reset", methods=["POST"])
def reset():
    """開発/検証用のリセット。本番では ENABLE_RESET_ROUTE=1 のときだけ有効。"""
    if os.getenv("ENABLE_RESET_ROUTE", "0") != "1":
        abort(404)

    user = get_current_user()
    if user:
        DiagnosisResult.query.filter_by(user_id=user.id).delete()
        JournalEntry.query.filter_by(user_id=user.id).delete()
        KaiLog.query.filter_by(user_id=user.id).delete()
        KaiAnalysisLog.query.filter_by(user_id=user.id).delete()
        QuestProgress.query.filter_by(user_id=user.id).delete()
        db.session.commit()
    session.clear()
    return redirect(url_for("index"))


@app.cli.command("init-db")
def init_db():
    """flask init-db"""
    with app.app_context():
        db.create_all()
        ensure_extra_columns()
    logger.info("DB initialized")


@app.route("/feedback")
def feedback():
    return redirect("https://docs.google.com/forms/d/e/1FAIpQLSfQaOZnQ-vMVPAjjuQtcFiuhTHz9eoHzLQOsISBkBd3Qm6rAA/viewform?usp=publish-editor")


if __name__ == "__main__":
    # ローカル起動用。本番Renderでは gunicorn app:app で起動する想定。
    if os.getenv("AUTO_CREATE_DB", "0") == "1":
        with app.app_context():
            db.create_all()
            ensure_extra_columns()

    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
    )
