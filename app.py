import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
import tempfile

import imageio_ffmpeg
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import streamlit as st


ALLOWED_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v"}
ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-m4v",
}
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_TRANSCRIPTION_SIZE_MB = 25
MAX_TRANSCRIPTION_SIZE_BYTES = MAX_TRANSCRIPTION_SIZE_MB * 1024 * 1024

# Local .env takes precedence over a stale system environment variable.
load_dotenv(Path(__file__).with_name(".env"), override=True)


def extract_audio(video_data: bytes, extension: str) -> bytes:
    """Extract an MP3 audio track from video bytes using bundled FFmpeg."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"input{extension}"
        output_path = Path(temp_dir) / "audio.mp3"
        input_path.write_bytes(video_data)

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(output_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=300,
        )
        if result.returncode != 0 or not output_path.exists():
            raise RuntimeError("Nie udało się znaleźć lub przetworzyć ścieżki audio.")

        return output_path.read_bytes()


def transcribe_to_vtt(audio_data: bytes, api_key: str) -> str:
    """Generate timestamped WebVTT subtitles with OpenAI Speech-to-Text."""
    audio_file = BytesIO(audio_data)
    audio_file.name = "audio.mp3"
    client = OpenAI(api_key=api_key)
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="vtt",
    )
    return str(transcription)


def vtt_to_plain_text(vtt: str) -> str:
    """Remove WebVTT metadata and timestamps for readable transcript display."""
    text_lines: list[str] = []
    for line in vtt.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT" or "-->" in stripped:
            continue
        if re.fullmatch(r"\d+", stripped):
            continue
        text_lines.append(stripped)
    return "\n".join(text_lines)


def parse_vtt(vtt: str) -> list[dict[str, str]]:
    """Convert WebVTT content into editable subtitle rows."""
    lines = vtt.replace("\r\n", "\n").split("\n")
    cues: list[dict[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if "-->" not in line:
            index += 1
            continue

        start, end = (part.strip().split()[0] for part in line.split("-->", maxsplit=1))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index].strip())
            index += 1
        cues.append({"Początek": start, "Koniec": end, "Tekst": "\n".join(text_lines)})
    return cues


def timestamp_seconds(value: str) -> float:
    """Validate a VTT timestamp and return its value in seconds."""
    normalized = value.strip().replace(",", ".")
    parts = normalized.split(":")
    if len(parts) not in (2, 3) or not re.fullmatch(r"\d{2}\.\d{3}", parts[-1]):
        raise ValueError(f"Niepoprawny czas: {value}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"Niepoprawny czas: {value}") from error
    if numbers[-1] >= 60 or numbers[-2] >= 60:
        raise ValueError(f"Niepoprawny czas: {value}")
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def cues_to_vtt(cues: list[dict[str, str]]) -> str:
    """Validate edited rows and rebuild a WebVTT subtitle file."""
    blocks = ["WEBVTT"]
    for number, cue in enumerate(cues, start=1):
        start = str(cue.get("Początek", "")).strip()
        end = str(cue.get("Koniec", "")).strip()
        text = str(cue.get("Tekst", "")).strip()
        if not start and not end and not text:
            continue
        if not text:
            raise ValueError(f"Wiersz {number}: tekst napisu jest pusty.")
        if timestamp_seconds(end) <= timestamp_seconds(start):
            raise ValueError(f"Wiersz {number}: koniec musi być później niż początek.")
        blocks.append(f"{start.replace(',', '.')} --> {end.replace(',', '.')}\n{text}")
    if len(blocks) == 1:
        raise ValueError("Napisy nie zawierają żadnych wierszy.")
    return "\n\n".join(blocks) + "\n"


def configured_api_key() -> str:
    """Read the API key from the environment or Streamlit secrets."""
    if key := os.getenv("OPENAI_API_KEY"):
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", ""))
    except (FileNotFoundError, KeyError):
        return ""


st.set_page_config(page_title="Odtwarzacz wideo", page_icon="🎬", layout="centered")

st.title("🎬 Odtwarzacz wideo")
st.write("Prześlij plik wideo, aby odtworzyć go bezpośrednio w przeglądarce.")

uploaded_file = st.file_uploader(
    "Wybierz plik wideo",
    type=[extension.removeprefix(".") for extension in sorted(ALLOWED_EXTENSIONS)],
    help=f"Obsługiwane formaty: MP4, WebM, MOV i M4V. Maksymalnie {MAX_FILE_SIZE_MB} MB.",
)

if uploaded_file is not None:
    extension = Path(uploaded_file.name).suffix.lower()
    mime_type = (uploaded_file.type or "").lower()
    file_size = uploaded_file.size

    errors: list[str] = []
    if extension not in ALLOWED_EXTENSIONS:
        errors.append("Nieobsługiwane rozszerzenie pliku.")
    if mime_type not in ALLOWED_MIME_TYPES:
        errors.append(f"Nieobsługiwany typ MIME: {mime_type or 'brak'}.")
    if file_size == 0:
        errors.append("Przesłany plik jest pusty.")
    if file_size > MAX_FILE_SIZE_BYTES:
        errors.append(f"Plik przekracza limit {MAX_FILE_SIZE_MB} MB.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        video_data = uploaded_file.getvalue()
        video_id = hashlib.sha256(video_data).hexdigest()
        if st.session_state.get("video_id") != video_id:
            previous_video_id = st.session_state.get("video_id")
            if previous_video_id:
                st.session_state.pop(f"subtitle_editor_{previous_video_id}", None)
            st.session_state.video_id = video_id
            st.session_state.pop("audio_data", None)
            st.session_state.pop("subtitles_vtt", None)

        st.success(f"Wczytano: {uploaded_file.name} ({file_size / 1024 / 1024:.1f} MB)")
        subtitles_vtt = st.session_state.get("subtitles_vtt")
        st.video(
            video_data,
            format=mime_type,
            subtitles=subtitles_vtt.encode("utf-8") if subtitles_vtt else None,
        )
        st.caption(
            "Jeśli film się nie odtwarza, jego kodek może nie być obsługiwany "
            "przez przeglądarkę. Najbardziej zgodny jest MP4 z kodekiem H.264."
        )

        st.subheader("Ścieżka dźwiękowa")
        if st.button("Wyodrębnij dźwięk", type="primary"):
            try:
                with st.spinner("Wyodrębnianie dźwięku…"):
                    st.session_state.audio_data = extract_audio(video_data, extension)
            except subprocess.TimeoutExpired:
                st.error("Przetwarzanie trwało zbyt długo i zostało przerwane.")
            except (OSError, RuntimeError) as error:
                st.error(str(error))

        if audio_data := st.session_state.get("audio_data"):
            st.audio(audio_data, format="audio/mpeg")
            output_name = f"{Path(uploaded_file.name).stem}_audio.mp3"
            st.download_button(
                "Pobierz dźwięk MP3",
                data=audio_data,
                file_name=output_name,
                mime="audio/mpeg",
            )

            st.subheader("Napisy")
            api_key = configured_api_key()
            if not api_key:
                api_key = st.text_input(
                    "Klucz OpenAI API",
                    type="password",
                    help="Klucz jest używany tylko do tego żądania i nie jest zapisywany w kodzie.",
                )

            if len(audio_data) >= MAX_TRANSCRIPTION_SIZE_BYTES:
                st.warning(
                    f"Plik audio ma {len(audio_data) / 1024 / 1024:.1f} MB. "
                    f"API transkrypcji przyjmuje pliki mniejsze niż {MAX_TRANSCRIPTION_SIZE_MB} MB."
                )
            elif st.button("Wygeneruj napisy", disabled=not bool(api_key)):
                try:
                    with st.spinner("Generowanie napisów…"):
                        st.session_state.subtitles_vtt = transcribe_to_vtt(audio_data, api_key)
                    st.session_state.pop(f"subtitle_editor_{video_id}", None)
                    st.rerun()
                except OpenAIError as error:
                    st.error(f"Błąd OpenAI API: {error}")

        if subtitles_vtt := st.session_state.get("subtitles_vtt"):
            st.markdown("#### Edycja napisów")
            st.caption(
                "Popraw tekst lub czasy wyświetlania. Format czasu: "
                "`MM:SS.mmm` albo `HH:MM:SS.mmm`. Możesz również dodawać i usuwać wiersze."
            )
            subtitle_cues = parse_vtt(subtitles_vtt)
            editor_key = f"subtitle_editor_{video_id}"
            edited_cues = st.data_editor(
                subtitle_cues,
                key=editor_key,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                column_config={
                    "Początek": st.column_config.TextColumn("Początek", width="small"),
                    "Koniec": st.column_config.TextColumn("Koniec", width="small"),
                    "Tekst": st.column_config.TextColumn("Tekst", width="large"),
                },
            )
            if st.button("Zapisz zmiany w napisach", type="primary"):
                rows = (
                    edited_cues.to_dict("records")
                    if hasattr(edited_cues, "to_dict")
                    else list(edited_cues)
                )
                try:
                    st.session_state.subtitles_vtt = cues_to_vtt(rows)
                    st.session_state.pop(editor_key, None)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

            transcript_text = vtt_to_plain_text(subtitles_vtt)
            with st.expander("Podgląd całej transkrypcji"):
                st.text_area("Treść napisów", transcript_text, height=240, disabled=True)
            subtitle_name = f"{Path(uploaded_file.name).stem}_napisy.vtt"
            transcript_name = f"{Path(uploaded_file.name).stem}_transkrypcja.txt"
            download_vtt, download_txt = st.columns(2)
            download_vtt.download_button(
                "Pobierz napisy VTT",
                data=subtitles_vtt.encode("utf-8"),
                file_name=subtitle_name,
                mime="text/vtt",
            )
            download_txt.download_button(
                "Pobierz transkrypcję TXT",
                data=transcript_text.encode("utf-8"),
                file_name=transcript_name,
                mime="text/plain",
            )
