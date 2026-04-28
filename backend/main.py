from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import anthropic
import os
import re

app = FastAPI(title="YouTube 한글 자막 번역기")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class TranslateRequest(BaseModel):
    url: str


def extract_video_id(url: str) -> str:
    """YouTube URL에서 video ID 추출"""
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("YouTube 영상 ID를 찾을 수 없습니다.")


def extract_subtitles_transcript_api(video_id: str):
    """youtube-transcript-api (v1.x) 로 자막 추출"""
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound, TranscriptsDisabled,
        VideoUnavailable, AgeRestricted, IpBlocked,
        PoTokenRequired, RequestBlocked, VideoUnplayable,
    )
    import httpx

    http_client = httpx.Client(timeout=15.0)
    api = YouTubeTranscriptApi(http_client=http_client)

    # 1) 영어 자막 직접 fetch
    for lang in [["en", "en-US", "en-GB"], ["en"]]:
        try:
            fetched = api.fetch(video_id, languages=lang)
            text = " ".join(snippet.text for snippet in fetched)
            if text.strip():
                return text, "en"
        except (NoTranscriptFound, Exception):
            continue

    # 2) TranscriptList에서 찾기
    try:
        transcript_list = api.list(video_id)
        # 수동 영어 자막 우선
        for t in transcript_list:
            if t.language_code.startswith("en") and not t.is_generated:
                fetched = t.fetch()
                text = " ".join(snippet.text for snippet in fetched)
                if text.strip():
                    return text, t.language_code
        # 자동 생성 영어
        for t in transcript_list:
            if t.language_code.startswith("en"):
                fetched = t.fetch()
                text = " ".join(snippet.text for snippet in fetched)
                if text.strip():
                    return text, t.language_code
        # 아무 언어나
        for t in transcript_list:
            fetched = t.fetch()
            text = " ".join(snippet.text for snippet in fetched)
            if text.strip():
                return text, t.language_code
    except (AgeRestricted,):
        raise ValueError("AGE_RESTRICTED")
    except (IpBlocked, RequestBlocked):
        raise ValueError("IP_BLOCKED")
    except (PoTokenRequired,):
        raise ValueError("IP_BLOCKED")
    except TranscriptsDisabled:
        raise ValueError("NO_SUBTITLES")
    except VideoUnavailable:
        raise ValueError("VIDEO_UNAVAILABLE")
    except Exception as e:
        err = str(e).lower()
        if "age" in err:
            raise ValueError("AGE_RESTRICTED")
        if "ip" in err or "blocked" in err or "robot" in err:
            raise ValueError("IP_BLOCKED")
        raise ValueError(f"transcript_api_error: {str(e)}")

    raise ValueError("NO_SUBTITLES")


def extract_subtitles_yt_dlp(url: str):
    """yt-dlp로 YouTube 자막 추출 (fallback)"""
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "ko", "ja", "zh"],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,
        "extractor_args": {
            "youtube": {"player_client": ["web", "mweb"]}
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        subtitles = info.get("subtitles", {})
        auto_subtitles = info.get("automatic_captions", {})

        sub_data = None
        lang_used = None

        for lang in ["en", "en-US", "en-GB"]:
            if lang in subtitles:
                sub_data = subtitles[lang]
                lang_used = lang
                break

        if not sub_data:
            for lang in ["en", "en-US", "en-GB"]:
                if lang in auto_subtitles:
                    sub_data = auto_subtitles[lang]
                    lang_used = lang
                    break

        if not sub_data:
            all_subs = {**subtitles, **auto_subtitles}
            if all_subs:
                lang_used = list(all_subs.keys())[0]
                sub_data = all_subs[lang_used]

        if not sub_data:
            raise ValueError("NO_SUBTITLES")

        vtt_url = None
        for fmt in sub_data:
            if fmt.get("ext") == "vtt":
                vtt_url = fmt.get("url")
                break
        if not vtt_url and sub_data:
            vtt_url = sub_data[0].get("url")

        if not vtt_url:
            raise ValueError("NO_SUBTITLES")

        import urllib.request
        with urllib.request.urlopen(vtt_url) as response:
            vtt_content = response.read().decode("utf-8")

        return parse_vtt(vtt_content), info.get("title", ""), lang_used


def parse_vtt(vtt_content: str) -> str:
    lines = vtt_content.split("\n")
    texts = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if text and text not in texts[-1:]:
            texts.append(text)
    return " ".join(texts)


def get_video_title(url: str) -> str:
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "")
    except Exception:
        return ""


def translate_with_claude(text: str, title: str) -> dict:
    max_chunk = 3000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    translated_parts = []

    for chunk in chunks:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""다음은 YouTube 영상 "{title}"의 자막입니다.
이 자막을 자연스럽고 완벽한 한국어로 번역해주세요.

규칙:
- 원문의 의미를 정확하게 전달
- 자연스러운 한국어 표현 사용
- 구어체 유지 (말하는 것처럼)
- 번역문만 출력 (설명 없이)

자막 원문:
{chunk}"""
            }]
        )
        translated_parts.append(message.content[0].text)

    return {
        "title": title,
        "translated_text": "\n\n".join(translated_parts),
        "original_length": len(text),
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube 한글 자막 번역기 API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/translate")
async def translate(req: TranslateRequest):
    url = req.url.strip()

    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="올바른 YouTube URL을 입력해주세요.")

    subtitle_text = None
    title = ""
    lang = "en"
    last_error = ""

    # 방법 1: youtube-transcript-api
    try:
        video_id = extract_video_id(url)
        subtitle_text, lang = extract_subtitles_transcript_api(video_id)
        title = get_video_title(url)
    except ValueError as e:
        last_error = str(e)
        subtitle_text = None
    except Exception as e:
        last_error = str(e)
        subtitle_text = None

    # 방법 2: yt-dlp fallback
    if not subtitle_text:
        try:
            subtitle_text, title, lang = extract_subtitles_yt_dlp(url)
        except Exception as e:
            if not last_error:
                last_error = str(e)
            subtitle_text = None

    # 에러 처리
    if not subtitle_text or len(subtitle_text) < 10:
        if last_error == "AGE_RESTRICTED":
            raise HTTPException(status_code=422, detail="연령 제한 영상입니다. 자막을 가져올 수 없습니다.")
        elif last_error == "IP_BLOCKED":
            raise HTTPException(status_code=422, detail="이 영상의 자막은 서버 IP에서 접근이 제한되어 있습니다. 다른 영상을 시도해보세요.")
        elif last_error == "NO_SUBTITLES":
            raise HTTPException(status_code=422, detail="이 영상에는 자막이 없습니다. 자막이 있는 영상만 번역 가능합니다.")
        elif last_error == "VIDEO_UNAVAILABLE":
            raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다. URL을 확인해주세요.")
        elif "sign in" in last_error.lower() or "login" in last_error.lower():
            raise HTTPException(status_code=422, detail="이 영상은 로그인이 필요합니다 (연령 제한 또는 멤버십 전용 영상).")
        elif "private" in last_error.lower():
            raise HTTPException(status_code=403, detail="비공개 영상입니다.")
        else:
            raise HTTPException(status_code=422, detail="자막을 가져올 수 없습니다. 자막이 있는 공개 영상 URL을 입력해주세요.")

    try:
        result = translate_with_claude(subtitle_text, title)
        return {
            "success": True,
            "title": result["title"],
            "translated_text": result["translated_text"],
            "original_language": lang,
            "character_count": result["original_length"],
      