from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import anthropic
import os
import re
import subprocess
import sys
import json

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


# subprocess에서 실행할 자막 추출 스크립트
_TRANSCRIPT_SCRIPT = r"""
import sys, json
video_id = sys.argv[1]
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound, TranscriptsDisabled, VideoUnavailable,
        AgeRestricted, IpBlocked, PoTokenRequired, RequestBlocked
    )
    api = YouTubeTranscriptApi()

    # 영어 자막 직접 fetch
    for lang in [["en", "en-US", "en-GB"], ["en"]]:
        try:
            fetched = api.fetch(video_id, languages=lang)
            text = " ".join(s.text for s in fetched)
            if text.strip():
                print(json.dumps({"text": text, "lang": "en"}))
                sys.exit(0)
        except Exception:
            continue

    # TranscriptList에서 찾기
    try:
        tlist = api.list(video_id)
        candidates = []
        for t in tlist:
            candidates.append(t)
        # 영어 수동 우선
        for t in sorted(candidates, key=lambda x: (not x.language_code.startswith("en"), x.is_generated)):
            try:
                fetched = t.fetch()
                text = " ".join(s.text for s in fetched)
                if text.strip():
                    print(json.dumps({"text": text, "lang": t.language_code}))
                    sys.exit(0)
            except Exception:
                continue
    except AgeRestricted:
        print(json.dumps({"error": "AGE_RESTRICTED"})); sys.exit(1)
    except (IpBlocked, RequestBlocked, PoTokenRequired):
        print(json.dumps({"error": "IP_BLOCKED"})); sys.exit(1)
    except TranscriptsDisabled:
        print(json.dumps({"error": "NO_SUBTITLES"})); sys.exit(1)
    except VideoUnavailable:
        print(json.dumps({"error": "VIDEO_UNAVAILABLE"})); sys.exit(1)
    except Exception as e:
        err = str(e).lower()
        if "age" in err:
            print(json.dumps({"error": "AGE_RESTRICTED"})); sys.exit(1)
        if "ip" in err or "blocked" in err or "robot" in err:
            print(json.dumps({"error": "IP_BLOCKED"})); sys.exit(1)
        print(json.dumps({"error": str(e)})); sys.exit(1)

    print(json.dumps({"error": "NO_SUBTITLES"}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
"""


def extract_subtitles_transcript_api(video_id: str, timeout: int = 22):
    """subprocess로 자막 추출 (타임아웃 시 프로세스 강제 종료 가능)"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _TRANSCRIPT_SCRIPT, video_id],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        if not stdout:
            stderr = result.stderr.strip()
            raise ValueError(f"subprocess_error: {stderr[:300]}")
        data = json.loads(stdout)
        if "error" in data:
            raise ValueError(data["error"])
        return data["text"], data["lang"]
    except subprocess.TimeoutExpired:
        raise ValueError("IP_BLOCKED")


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
        with urllib.request.urlopen(vtt_url, timeout=15) as response:
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
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "socket_timeout": 10}) as ydl:
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

    # 방법 1: youtube-transcript-api (subprocess 22초 타임아웃 - 프로세스 강제 종료 가능)
    try:
        video_id = extract_video_id(url)
        subtitle_text, lang = extract_subtitles_transcript_api(video_id, timeout=22)
        try:
            title = get_video_title(url)
        except Exception:
            title = ""
    except ValueError as e:
        last_error = str(e)
        subtitle_text = None
    except Exception as e:
        last_error = str(e)
        subtitle_text = None

    # 방법 2: yt-dlp fallback (socket_timeout=15으로 자체 처리)
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
     