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


def extract_subtitles_yt_dlp(url: str) -> str:
    """yt-dlp로 YouTube 자막 추출 (영어 우선, 자동 생성 포함)"""
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "ko", "ja", "zh"],
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
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
            raise ValueError("이 영상에는 자막이 없습니다.")

        vtt_url = None
        for fmt in sub_data:
            if fmt.get("ext") == "vtt":
                vtt_url = fmt.get("url")
                break
        if not vtt_url and sub_data:
            vtt_url = sub_data[0].get("url")

        if not vtt_url:
            raise ValueError("자막 URL을 찾을 수 없습니다.")

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
                "content": f'다음은 YouTube 영상 "{title}"의 자막입니다.\n자연스럽고 완벽한 한국어로 번역해주세요.\n규칙:\n- 원문의 의미를 정확하게 전달\n- 자연스러운 한국어 표현 사용\n- 구어체 유지\n- 번역문만 출력\n\n자막 원문:\n' + chunk
            }]
        )
        translated_parts.append(message.content[0].text)
    return {"title": title, "translated_text": "\n\n".join(translated_parts), "original_length": len(text)}


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
    try:
        subtitle_text, title, lang = extract_subtitles_yt_dlp(url)
        if not subtitle_text or len(subtitle_text) < 10:
            raise HTTPException(status_code=422, detail="자막 내용이 너무 짧거나 비어있습니다.")
        result = translate_with_claude(subtitle_text, title)
        return {"success": True, "title": result["title"], "translated_text": result["translated_text"], "original_language": lang, "character_count": result["original_length"]}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        if "Video unavailable" in error_msg:
            raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
        elif "Private video" in error_msg:
            raise HTTPException(status_code=403, detail="비공개 영상입니다.")
        elif "subtitles" in error_msg.lower() or "caption" in error_msg.lower():
            raise HTTPException(status_code=422, detail="이 영상에는 자막이 없습니다.")
        else:
            raise HTTPException(status_code=500, detail=f"오류: {error_msg}")
