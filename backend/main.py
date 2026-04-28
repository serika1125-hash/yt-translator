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
    raise ValueError("YouTube ID not found")


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
    for lang in [["en", "en-US", "en-GB"], ["en"]]:
        try:
            fetched = api.fetch(video_id, languages=lang)
            text = " ".join(s.text for s in fetched)
            if text.strip():
                print(json.dumps({"text": text, "lang": "en"}))
                sys.exit(0)
        except Exception:
            continue
    try:
        tlist = api.list(video_id)
        candidates = list(tlist)
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
    print(json.dumps({"error": "NO_SUBTITLES"})); sys.exit(1)
except Exception as e:
    print(json.dumps({"error": str(e)})); sys.exit(1)
"""


def extract_subtitles_transcript_api(video_id: str, timeout: int = 22):
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
            raise ValueError("subprocess_error: " + stderr[:300])
        data = json.loads(stdout)
        if "error" in data:
            raise ValueError(data["error"])
        return data["text"], data["lang"]
    except subprocess.TimeoutExpired:
        raise ValueError("IP_BLOCKED")


def extract_subtitles_yt_dlp(url: str):
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "ko", "ja", "zh"],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 15,
        "extractor_args": {"youtube": {"player_client": ["web", "mweb"]}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subtitles = info.get("subtitles", {})
        auto_subtitles = info.get("automatic_captions", {})
        sub_data = None
        lang_used = None
        for lang in ["en", "en-US", "en-GB"]:
            if lang in subtitles:
                sub_data = subtitles[lang]; lang_used = lang; break
        if not sub_data:
            for lang in ["en", "en-US", "en-GB"]:
                if lang in auto_subtitles:
                    sub_data = auto_subtitles[lang]; lang_used = lang; break
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
                vtt_url = fmt.get("url"); break
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
        opts = {"quiet": True, "no_warnings": True, "skip_download": True, "socket_timeout": 10}
        with yt_dlp.YoutubeDL(opts) as ydl:
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
                "content": (
                    'YouTube video "' + title + '" subtitles below.\n'
                    'Translate to natural Korean. Output translation only.\n\n'
                    + chunk
                )
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
    return {"status": "ok", "message": "YouTube Korean Subtitle Translator API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/translate")
async def translate(req: TranslateRequest):
    url = req.url.strip()
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="Please enter a valid YouTube URL.")

    subtitle_text = None
    title = ""
    lang = "en"
    last_error = ""

    # Method 1: youtube-transcript-api via subprocess (22s hard timeout)
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

    # Method 2: yt-dlp fallback
    if not subtitle_text:
        try:
            subtitle_text, title, lang = extract_subtitles_yt_dlp(url)
        except Exception as e:
            if not last_error:
                last_error = str(e)
            subtitle_text = None

    if not subtitle_text or len(subtitle_text) < 10:
        if last_error == "AGE_RESTRICTED":
            raise HTTPException(status_code=422, detail="Age-restricted video. Cannot fetch subtitles.")
        elif last_error == "IP_BLOCKED":
            raise HTTPException(status_code=422, detail="Server IP blocked by YouTube. Try a different video.")
        elif last_error == "NO_SUBTITLES":
            raise HTTPException(status_code=422, detail="No subtitles found. Only videos with subtitles can be translated.")
        elif last_error == "VIDEO_UNAVAILABLE":
            raise HTTPException(status_code=404, detail="Video not found. Check the URL.")
        elif "sign in" in last_error.lower() or "login" in last_error.lower():
            raise HTTPException(status_code=422, detail="Login required (age-restricted or members-only video).")
        elif "private" in last_error.lower():
            raise HTTPException(status_code=403, detail="Private video.")
        else:
            raise HTTPException(status_code=422, detail="Cannot fetch subtitles. Use a public video with subtitles.")

    try:
        result = translate_with_claude(subtitle_text, title)
        return {
            "success": True,
            "title": result["title"],
            "translated_text": result["translated_text"],
            "original_language": lang,
            "character_count": result["original_length"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Translation error: " + str(e))
