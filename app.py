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


st.set_page_config(page_title="Subtitle Studio", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #000000; color: #ffffff; }
    .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }
    .stApp p, .stApp label, .stApp li, .stApp span,
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
    [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stSidebar"] { background: #050505; border-right: 1px solid #2f2f2f; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    .hero {
        padding: 2.2rem 2.4rem; border: 1px solid #7c3aed;
        border-radius: 24px; margin-bottom: 1.5rem;
        background: linear-gradient(125deg, #190b35, #090909 58%, #101b39);
        box-shadow: 0 24px 70px rgba(0,0,0,.28);
    }
    .hero h1 { color: #ffffff !important; margin: 0; font-size: 2.8rem; letter-spacing: -.04em; }
    .hero p { color: #ffffff !important; margin: .65rem 0 0; font-size: 1.05rem; }
    .eyebrow { color: #c4b5fd !important; font-size: .78rem; font-weight: 800; letter-spacing: .14em; }
    [data-testid="stImage"] img { border-radius: 22px; border: 1px solid #343434; }
    .step-card {
        min-height: 92px; padding: 1rem 1.05rem; border-radius: 16px;
        color: #ffffff; border: 1px solid #454545; background: #0c0c0c;
    }
    .step-card.done { border-color: #34d399; background: #06291f; }
    .step-number { color: #e2e8f0 !important; font-size: .72rem; font-weight: 700; }
    .step-title { color: #ffffff !important; margin-top: .35rem; font-weight: 750; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #090909; border-color: #3b3b3b;
        border-radius: 18px; box-shadow: 0 14px 35px rgba(0,0,0,.16);
    }
    [data-testid="stFileUploaderDropzone"] {
        color: #ffffff; background: #0d0d0d; border-radius: 16px; border-color: #a78bfa;
    }
    [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] {
        color: #ffffff !important; background: #050505 !important; border-color: #555555 !important;
    }
    [data-testid="stDataFrame"] { border: 1px solid #454545; border-radius: 12px; }
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFileUploader"] button {
        color: #ffffff !important;
        background: linear-gradient(90deg, #4c1d95, #1e3a8a) !important;
        border: 1px solid #8b5cf6 !important; border-radius: 12px; font-weight: 700;
        transition: background-color .18s ease, border-color .18s ease, transform .18s ease;
    }
    .stButton > button *, .stDownloadButton > button *,
    [data-testid="stFileUploader"] button * { color: #ffffff !important; }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFileUploader"] button:hover {
        color: #ffffff !important;
        background: linear-gradient(90deg, #6d28d9, #1d4ed8) !important;
        border-color: #c4b5fd !important; transform: translateY(-1px);
    }
    .stButton > button:focus, .stDownloadButton > button:focus,
    [data-testid="stFileUploader"] button:focus {
        color: #ffffff !important;
        background: linear-gradient(90deg, #5b21b6, #1e40af) !important;
        border-color: #ddd6fe !important; box-shadow: 0 0 0 2px rgba(167,139,250,.35) !important;
    }
    .stButton > button:disabled {
        color: #d1d5db !important; background: #181818 !important;
        border-color: #404040 !important; opacity: .75;
    }
    .stButton > button[kind="primary"] {
        border: 1px solid #a78bfa !important;
        box-shadow: 0 8px 24px rgba(99,102,241,.28);
    }
    [data-testid="stMetric"] { background: #111111; border: 1px solid #333333; padding: .7rem; border-radius: 12px; }
    </style>
    <div class="hero">
      <div class="eyebrow">AI VIDEO WORKSPACE</div>
      <h1>🎬 Subtitle Studio</h1>
      <p>Wczytaj film, wyodrębnij dźwięk, wygeneruj napisy i dopracuj je w jednym miejscu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_image = Path(__file__).with_name("assets") / "cinema-hero.png"
if hero_image.exists():
    st.image(str(hero_image), width="stretch")

with st.sidebar:
    st.title("Subtitle Studio")
    st.caption("Generator i edytor napisów")
    st.divider()
    if configured_api_key():
        st.success("Klucz OpenAI API wykryty")
    else:
        st.warning("Brak klucza OpenAI API")
    st.markdown("**Obsługiwane formaty**")
    st.caption("MP4 · WebM · MOV · M4V")
    st.markdown("**Limit filmu**")
    st.caption(f"{MAX_FILE_SIZE_MB} MB")
    st.divider()
    st.caption("🔒 Klucz API pozostaje po stronie aplikacji.")

st.subheader("1. Wczytaj materiał")
st.caption("Wybierz film, od którego chcesz rozpocząć pracę.")

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

        subtitles_vtt = st.session_state.get("subtitles_vtt")
        audio_data = st.session_state.get("audio_data")
        steps = [
            ("01", "Wideo wczytane", True),
            ("02", "Dźwięk gotowy", bool(audio_data)),
            ("03", "Napisy wygenerowane", bool(subtitles_vtt)),
            ("04", "Edycja i eksport", bool(subtitles_vtt)),
        ]
        step_columns = st.columns(4)
        for column, (number, title, done) in zip(step_columns, steps):
            state_class = " done" if done else ""
            marker = "✓" if done else "○"
            column.markdown(
                f'<div class="step-card{state_class}"><div class="step-number">ETAP {number}</div>'
                f'<div class="step-title">{marker} {title}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        media_column, tools_column = st.columns([1.55, 1], gap="large")
        with media_column:
            with st.container(border=True):
                st.markdown("### Podgląd filmu")
                st.video(
                    video_data,
                    format=mime_type,
                    subtitles=subtitles_vtt.encode("utf-8") if subtitles_vtt else None,
                )
                st.caption(
                    "Jeśli film się nie odtwarza, jego kodek może nie być obsługiwany "
                    "przez przeglądarkę. Najbardziej zgodny jest MP4/H.264."
                )

        with tools_column:
            with st.container(border=True):
                st.markdown("### Materiał")
                st.caption(uploaded_file.name)
                metric_format, metric_size = st.columns(2)
                metric_format.metric("Format", extension.removeprefix(".").upper())
                metric_size.metric("Rozmiar", f"{file_size / 1024 / 1024:.1f} MB")

                st.divider()
                st.markdown("### 2. Ścieżka dźwiękowa")
                if st.button("🎵 Wyodrębnij dźwięk", type="primary", width="stretch"):
                    try:
                        with st.spinner("Wyodrębnianie dźwięku…"):
                            st.session_state.audio_data = extract_audio(video_data, extension)
                        st.rerun()
                    except subprocess.TimeoutExpired:
                        st.error("Przetwarzanie trwało zbyt długo i zostało przerwane.")
                    except (OSError, RuntimeError) as error:
                        st.error(str(error))

                if audio_data := st.session_state.get("audio_data"):
                    st.audio(audio_data, format="audio/mpeg")
                    output_name = f"{Path(uploaded_file.name).stem}_audio.mp3"
                    st.download_button(
                        "↓ Pobierz MP3",
                        data=audio_data,
                        file_name=output_name,
                        mime="audio/mpeg",
                        width="stretch",
                    )

                    st.divider()
                    st.markdown("### 3. Generowanie napisów")
                    api_key = configured_api_key()
                    if not api_key:
                        api_key = st.text_input(
                            "Klucz OpenAI API",
                            type="password",
                            help="Klucz jest używany tylko do tego żądania i nie jest zapisywany w kodzie.",
                        )

                    if len(audio_data) >= MAX_TRANSCRIPTION_SIZE_BYTES:
                        st.warning(
                            f"Audio ma {len(audio_data) / 1024 / 1024:.1f} MB. "
                            f"Limit API wynosi {MAX_TRANSCRIPTION_SIZE_MB} MB."
                        )
                    elif st.button(
                        "✨ Wygeneruj napisy",
                        disabled=not bool(api_key),
                        width="stretch",
                    ):
                        try:
                            with st.spinner("Generowanie napisów…"):
                                st.session_state.subtitles_vtt = transcribe_to_vtt(audio_data, api_key)
                            st.session_state.pop(f"subtitle_editor_{video_id}", None)
                            st.rerun()
                        except OpenAIError as error:
                            st.error(f"Błąd OpenAI API: {error}")

        if subtitles_vtt := st.session_state.get("subtitles_vtt"):
            st.divider()
            st.markdown("## 4. Edycja i eksport")
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
                "↓ Pobierz napisy VTT",
                data=subtitles_vtt.encode("utf-8"),
                file_name=subtitle_name,
                mime="text/vtt",
                width="stretch",
            )
            download_txt.download_button(
                "↓ Pobierz transkrypcję TXT",
                data=transcript_text.encode("utf-8"),
                file_name=transcript_name,
                mime="text/plain",
                width="stretch",
            )
