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
    """youtube-transcript-api로 자막 추출 (로그인 불필요)"""
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

    # 언어 우선순위: 영어 > 자동생성 영어 > 나머지
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except TranscriptsDisabled:
        raise ValueError("이 영상에는 자막이 비활성화되어 있습니다.")
    except Exception as e:
        raise ValueError(f"자막 목록을 불러올 수 없습니다: {str(e)}")

    transcript = None
    lang_used = None

    # 1) 수동 영어 자막
    try:
        transcript = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"])
        lang_used = "en"
    except NoTranscriptFound:
        pass

    # 2) 자동생성 영어 자막
    if not transcript:
        try:
            transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])
            lang_used = "en (auto)"
        except NoTranscriptFound:
            pass

    # 3) 한국어 자막 (이미 한국어면 그냥 반환)
    if not transcript:
        try:
            transcript = transcript_list.find_manually_created_transcript(["ko"])
            lang_used = "ko"
        except NoTranscriptFound:
            pass

    # 4) 아무 언어나
    if not transcript:
        try:
            all_transcripts = list(transcript_list)
            if all_transcripts:
                transcript = all_transcripts[0]
                lang_used = transcript.language_code
        except Exception:
            pass

    if not transcript:
        raise ValueError("이 영상에는 사용 가능한 자막이 없습니다.")

    data = transcript.fetch()
    text = " ".join([entry["text"] for entry in data])
    return text, lang_used


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
    """VTT 자막 파일에서 순수 텍스트만 추출"""
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

        text = re.sub(r"<[^>]+>", "", line)
        text = text.strip()

        if text and text not in texts[-1:]:
            texts.append(text)

    return " ".join(texts)


def get_video_title(url: str) -> str:
    """영상 제목만 가져오기"""
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title", "")
    except Exception:
        return ""


def translate_with_claude(text: str, title: str) -> dict:
    """Claude API로 한글 번역"""
    max_chunk = 3000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    translated_parts = []

    for chunk in chunks:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[
                {
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
                }
            ]
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

    # 방법 1: youtube-transcript-api (로그인 불필요, 빠름)
    try:
        video_id = extract_video_id(url)
        subtitle_text, lang = extract_subtitles_transcript_api(video_id)
        title = get_video_title(url)
    except Exception as e:
        last_error = str(e)
        subtitle_text = None

    # 방법 2: yt-dlp fallback
    if not subtitle_text:
        try:
            subtitle_text, title, lang = extract_subtitles_yt_dlp(url)
        except Exception as e:
            last_error = str(e)
            subtitle_text = None

    if not subtitle_text or len(subtitle_text) < 10:
        # 에러 메시지 정리
        if "sign in" in last_error.lower() or "login" in last_error.lower():
            raise HTTPException(status_code=422, detail="이 영상은 로그인이 필요하거나 자막이 없습니다.")
        elif "disabled" in last_error.lower():
            raise HTTPException(status_code=422, detail="이 영상에는 자막이 비활성화되어 있습니다.")
        elif "unavailable" in last_error.lower():
            raise HTTPException(status_code=404, detail="영상을 찾을 수 없습니다.")
        elif "private" in last_error.lower():
            raise HTTPException(status_code=403, detail="비공개 영상입니다.")
        else:
            raise HTTPException(status_code=422, detail="자막을 가져올 수 없습니다. 자막이 없는 영상이거나 접근이 제한된 영상입니다.")

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
        raise HTTPException(status_code=500, detail=f"번역 중 오류: {str(e)}")
