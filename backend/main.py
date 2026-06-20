from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import yt_dlp
import os
import uuid

# Load variables from the .env file (this is where GROQ_API_KEY lives)
load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = FastAPI()

# Fallback: if cookies were provided via environment variable instead of
# Render's Secret Files feature, write them to disk so yt-dlp can use them.
_cookies_env = os.environ.get("YOUTUBE_COOKIES")
_cookie_path_at_startup = os.path.join(os.path.dirname(__file__), "cookies.txt")
if _cookies_env and not os.path.exists(_cookie_path_at_startup):
    with open(_cookie_path_at_startup, "w", encoding="utf-8") as f:
        f.write(_cookies_env)

# Allow only our deployed frontend to talk to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://youtubesummm.netlify.app",
        "http://127.0.0.1:5500",   # for local testing (e.g. VS Code Live Server)
        "null",                     # for opening index.html directly via file://
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder where downloaded audio files will be temporarily stored
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class VideoRequest(BaseModel):
    url: str
    mode: str = "summary"  # "summary" for quick overview, "notes" for full structured notes


def download_audio_from_youtube(video_url: str) -> dict:
    """
    Downloads audio from a YouTube URL and returns info about the file.
    Raises HTTPException on failure.
    """
    file_id = str(uuid.uuid4())
    output_path = os.path.join(DOWNLOAD_DIR, file_id)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    # If a cookies file is present, use it so requests look like they're
    # coming from a logged-in browser session (helps avoid YouTube's
    # bot-detection on cloud server IPs). On Render's Docker-based deploys,
    # secret files land at /etc/secrets/<filename>, not the app root.
    render_secret_path = "/etc/secrets/cookies.txt"
    local_cookie_path = os.path.join(os.path.dirname(__file__), "cookies.txt")

    if os.path.exists(render_secret_path):
        ydl_opts["cookiefile"] = render_secret_path
    elif os.path.exists(local_cookie_path):
        ydl_opts["cookiefile"] = local_cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            title = info.get("title", "Unknown title")
            duration = info.get("duration", 0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download audio: {str(e)}")

    final_file = f"{output_path}.mp3"

    if not os.path.exists(final_file):
        raise HTTPException(status_code=500, detail="Audio file was not created")

    return {
        "title": title,
        "duration_seconds": duration,
        "file_path": final_file,
    }


def transcribe_audio(file_path: str) -> str:
    """
    Sends an audio file to Groq's Whisper model and returns the transcript text.
    Raises HTTPException on failure.
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="text",
            )
        return transcription
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


def summarize_transcript(transcript: str, title: str) -> str:
    """
    Sends the transcript to a Groq-hosted LLM and asks for a short
    overview plus bullet-point key takeaways.
    """
    prompt = f"""You are summarizing a YouTube video transcript for someone who hasn't watched it.

Video title: {title}

Transcript:
{transcript}

Write your response in exactly this format:

**Overview:**
A short 2-4 sentence summary of what the video is about and its main point.

**Key Takeaways:**
- First key point
- Second key point
- (continue with as many bullet points as needed to cover the important ideas, typically 4-8)

Keep it clear and concise. Do not add any text before "**Overview:**" or after the last bullet point."""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


def generate_notes(transcript: str, title: str) -> str:
    """
    Sends the transcript to a Groq-hosted LLM and asks for long-form,
    structured study notes broken into sections with headers.
    """
    prompt = f"""You are turning a YouTube video transcript into thorough study notes for someone who wants to learn the material without watching the video.

Video title: {title}

Transcript:
{transcript}

Write detailed, well-organized notes using this structure:

## [Section Title]
Brief context for this section if helpful.
- Detailed point
- Detailed point
  - Sub-point if it adds useful detail

## [Next Section Title]
...continue this pattern...

Rules:
- Break the content into as many "## " sections as the material naturally calls for, named after what's actually discussed (not generic labels like "Section 1").
- Cover specific facts, numbers, examples, names, and steps mentioned in the transcript — don't just restate things vaguely.
- Use sub-bullets (indented with 2 spaces and a "-") for supporting detail under a main bullet.
- Write in plain, clear language a student could study from directly.
- Do not include any text before the first "## " heading or after the last bullet point. No closing remarks."""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notes generation failed: {str(e)}")


@app.get("/")
def root():
    return {"status": "Backend is running"}


@app.post("/download-audio")
def download_audio(request: VideoRequest):
    """
    Takes a YouTube URL, downloads just the audio, and returns
    the path + video title. Useful for testing the download step alone.
    """
    result = download_audio_from_youtube(request.url)
    file_size_mb = round(os.path.getsize(result["file_path"]) / (1024 * 1024), 2)
    result["file_size_mb"] = file_size_mb
    return result


@app.post("/transcribe")
def transcribe(request: VideoRequest):
    """
    Stage 2: Takes a YouTube URL, downloads the audio, transcribes it
    with Groq's Whisper model, and returns the full transcript text.
    Cleans up the audio file afterward since we don't need to keep it.
    """
    video_info = download_audio_from_youtube(request.url)
    file_path = video_info["file_path"]

    transcript_text = transcribe_audio(file_path)

    # Clean up the audio file now that we have the transcript
    try:
        os.remove(file_path)
    except OSError:
        pass  # not critical if cleanup fails

    return {
        "title": video_info["title"],
        "duration_seconds": video_info["duration_seconds"],
        "transcript": transcript_text,
    }


@app.post("/summarize")
def summarize(request: VideoRequest):
    """
    The full pipeline. Takes a YouTube URL, downloads the audio,
    transcribes it, then either summarizes it (mode="summary") or
    generates long-form structured notes (mode="notes").
    """
    video_info = download_audio_from_youtube(request.url)
    file_path = video_info["file_path"]

    transcript_text = transcribe_audio(file_path)

    # Clean up the audio file now that we have the transcript
    try:
        os.remove(file_path)
    except OSError:
        pass

    if request.mode == "notes":
        result_text = generate_notes(transcript_text, video_info["title"])
    else:
        result_text = summarize_transcript(transcript_text, video_info["title"])

    return {
        "title": video_info["title"],
        "duration_seconds": video_info["duration_seconds"],
        "transcript": transcript_text,
        "mode": request.mode,
        "summary": result_text,
    }