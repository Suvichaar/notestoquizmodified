# app.py — Notes/Quiz OCR → GPT → AMP Web Story
# Adds: quality tuning, subject depth, narration, SSML + (optional) Azure TTS, template validator, publisher meta, flexible S3 prefix.

import io
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as st_html  # inline HTML viewer
from PIL import Image

# Azure SDKs
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

# Optional Azure Speech (for TTS)
try:
    import azure.cognitiveservices.speech as speechsdk
    HAS_SPEECH = True
except Exception:
    HAS_SPEECH = False

# AWS S3
import boto3


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Notes → OCR → Quiz → AMP", page_icon="🧠", layout="centered")
st.title("🧠 Notes/Quiz OCR → GPT Structuring → AMP Web Story")
st.caption("Upload notes image(s) or a pre-made quiz image (or JSON), plus an AMP HTML template → download timestamped final HTML.")


# ---------------------------
# Secrets / Config
# ---------------------------
try:
    # Azure DI + OpenAI
    AZURE_DI_ENDPOINT = st.secrets["AZURE_DI_ENDPOINT"]
    AZURE_API_KEY = st.secrets["AZURE_API_KEY"]

    AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
    AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", AZURE_API_KEY)
    GPT_DEPLOYMENT = st.secrets.get("GPT_DEPLOYMENT", "gpt-4")

    # Optional Azure Speech for TTS
    AZURE_SPEECH_KEY = st.secrets.get("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION = st.secrets.get("AZURE_SPEECH_REGION", "")

    # AWS / S3
    AWS_ACCESS_KEY_ID     = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_REGION            = st.secrets.get("AWS_REGION", "ap-south-1")
    AWS_BUCKET            = st.secrets.get("AWS_BUCKET", "suvichaarapp")

    # Upload prefixes + CDN base
    # Default HTML prefix now empty (root), per your request (template fixing).
    HTML_S3_PREFIX = st.secrets.get("HTML_S3_PREFIX", "")
    AUDIO_S3_PREFIX = st.secrets.get("AUDIO_S3_PREFIX", "media/audio")
    CDN_HTML_BASE  = st.secrets.get("CDN_HTML_BASE", "https://stories.suvichaar.org/")
    CDN_MEDIA_BASE = st.secrets.get("CDN_MEDIA_BASE", CDN_HTML_BASE)  # reuse if not separate
except Exception:
    st.error("Missing secrets. Please set required Azure and AWS keys in .streamlit/secrets.toml")
    st.stop()


# ---------------------------
# Clients
# ---------------------------
di_client = DocumentIntelligenceClient(endpoint=AZURE_DI_ENDPOINT, credential=AzureKeyCredential(AZURE_API_KEY))
gpt_client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


# ---------------------------
# Utility: language + slug + placeholder validation
# ---------------------------
DEVANAGARI_RANGE = (0x0900, 0x097F)

def is_devanagari_char(ch: str) -> bool:
    cp = ord(ch)
    return DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]

def detect_lang(text: str) -> str:
    """Very light detector: if a fair share of chars are Devanagari -> hi-IN else en-US."""
    if not text:
        return "en-US"
    dev_count = sum(1 for ch in text if is_devanagari_char(ch))
    return "hi-IN" if dev_count >= max(10, len(text) * 0.05) else "en-US"

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s.lower()[:90]

# Required placeholders your AMP template should support (base set).
REQUIRED_KEYS = [
    "pagetitle", "storytitle", "typeofquiz", "potraitcoverurl",
    "s1title1", "s1text1",
    # Slides 2..6 (questions)
    *[f"s{n}{k}" for n in range(2, 7) for k in [
        "questionHeading", "question1",
        "option1", "option1attr", "option2", "option2attr",
        "option3", "option3attr", "option4", "option4attr",
        "attachment1"
    ]],
    "results_bg_image", "results_prompt_text", "results1_text", "results2_text", "results3_text",
]

# Optional audio/SSML placeholders (will be produced if you enable narration/SSML)
OPTIONAL_AUDIO_KEYS = [f"s{n}{k}" for n in range(1, 7) for k in ["narration", "ssml", "audio_url"]]

def validate_template_placeholders(template_html: str):
    missing = []
    for k in REQUIRED_KEYS:
        if f"{{{{{k}}}}}" not in template_html and f"{{{{{k}|safe}}}}" not in template_html:
            missing.append(k)
    # Report only missing required; optional listed separately for info.
    optional_present = [k for k in OPTIONAL_AUDIO_KEYS if (f"{{{{{k}}}}}" in template_html or f"{{{{{k}|safe}}}}" in template_html)]
    return missing, optional_present


# ---------------------------
# Prompts (base)
# ---------------------------
SYSTEM_PROMPT_OCR_TO_QA_BASE = """
You receive OCR text that already contains multiple-choice questions in Hindi or English.
Each question has options (A)-(D), a single correct answer, and ideally an explanation.

Return ONLY valid JSON:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "...",
      "narration": "Engaging, spoken-friendly 1–2 sentence narration for audio"
    },
    ...
  ]
}

- If explanations are missing, write a concise 1–2 sentence explanation grounded in the text.
- Preserve the original language (Hindi stays Hindi, English stays English).
"""

SYSTEM_PROMPT_NOTES_TO_QA_BASE = """
You are given raw study notes text (could be Hindi or English). Generate exactly FIVE high-quality,
multiple-choice questions (MCQs) strictly grounded in these notes.

For each question:
- Provide four options labeled A–D.
- Ensure exactly one correct option.
- Add a 1–2 sentence explanation that justifies the correct answer using the notes.
- Add a 1–2 sentence narration written for speech (engaging, human, no markup).

Respond ONLY with valid JSON:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "...",
      "narration": "..."
    },
    ...
  ]
}

Language: Use the same language as the notes. Keep questions concise, unambiguous, and correct.
"""

SYSTEM_PROMPT_QA_TO_PLACEHOLDERS_BASE = """
You're given:
{
  "questions": [
    {
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_option": "A"|"B"|"C"|"D",
      "explanation": "...",
      "narration": "..."   // optional
    }, ...
  ]
}

Create a flat JSON object with EXACTLY these keys (fill sensible defaults if missing).
Use the SAME language as input (Hindi stays Hindi):

pagetitle, storytitle, typeofquiz, potraitcoverurl,
s1title1, s1text1, s1narration, s1ssml, s1audio_url,

s2questionHeading, s2question1,
s2option1, s2option1attr, s2option2, s2option2attr,
s2option3, s2option3attr, s2option4, s2option4attr,
s2attachment1, s2narration, s2ssml, s2audio_url,

s3questionHeading, s3question1,
s3option1, s3option1attr, s3option2, s3option2attr,
s3option3, s3option3attr, s3option4, s3option4attr,
s3attachment1, s3narration, s3ssml, s3audio_url,

s4questionHeading, s4question1,
s4option1, s4option1attr, s4option2, s4option2attr,
s4option3, s4option3attr, s4option4, s4option4attr,
s4attachment1, s4narration, s4ssml, s4audio_url,

s5questionHeading, s5question1,
s5option1, s5option1attr, s5option2, s5option2attr,
s5option3, s5option3attr, s5option4, s5option4attr,
s5attachment1, s5narration, s5ssml, s5audio_url,

s6questionHeading, s6question1,
s6option1, s6option1attr, s6option2, s6option2attr,
s6option3, s6option3attr, s6option4, s6option4attr,
s6attachment1, s6narration, s6ssml, s6audio_url,

results_bg_image, results_prompt_text, results1_text, results2_text, results3_text

Mapping:
- We need FIVE Qs: map q[0]→s2*, q[1]→s3*, …, q[4]→s6*.
- sNquestion1 ← q[N-2].question (N=2..6)
- sNoption1..4 ← options A..D text
- For the correct option, set sNoptionKattr to "correct"; others "".
- sNattachment1 ← explanation
- sNnarration ← narration (or a concise spoken-friendly summary if missing)
- sNssml ← Empty now (a separate step will produce SSML; put "" placeholder)
- sNaudio_url ← "" (to be filled if TTS is generated)
- sNquestionHeading ← "Question {N-1}" (Hindi: "प्रश्न {N-1}")
- pagetitle/storytitle: short relevant titles from overall content.
- typeofquiz: "Educational" (Hindi: "शैक्षिक") if unknown.
- s1title1: 2–5 word intro; s1text1: 1–2 sentence intro; s1narration: engaging spoken intro; s1ssml / s1audio_url: "" for now.
- results_*: short friendly strings in same language. results_bg_image: "" if none.

Return only the JSON object.
""".strip()


# ---------------------------
# Dynamic prompt enrichment based on UI (quality + subject depth)
# ---------------------------
def enrich_with_quality_subject(base_prompt: str, subject: str, grade: str, subtopic: str,
                                depth_level: int, difficulty: str, bloom: str,
                                distractor_rigor: str, explanation_style: str,
                                examples_pref: str) -> str:
    addendum = f"""
Constraints:
- Subject: {subject or "General"}
- Grade: {grade or "Generic"}
- Subtopic: {subtopic or "N/A"}
- Depth level (1-5): {depth_level}
- Target difficulty: {difficulty}
- Cognitive emphasis (Bloom): {bloom}
- Distractors: {distractor_rigor} (avoid giveaways, plausible & disjoint)
- Explanations: {explanation_style}
- Examples preference: {examples_pref}
If source text quality is poor, still enforce clarity, correctness, and single-best-answer discipline.
"""
    return base_prompt.strip() + "\n\n" + addendum.strip()


# ---------------------------
# Helpers: OCR + GPT + JSON cleanup
# ---------------------------
def clean_model_json(txt: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", txt, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return txt.strip()

def ocr_extract(image_bytes: bytes) -> str:
    poller = di_client.begin_analyze_document(model_id="prebuilt-read", body=image_bytes)
    result = poller.result()
    if getattr(result, "paragraphs", None):
        return "\n".join([p.content for p in result.paragraphs]).strip()
    if getattr(result, "content", None):
        return result.content.strip()
    lines = []
    for page in getattr(result, "pages", []) or []:
        for line in getattr(page, "lines", []) or []:
            if getattr(line, "content", None):
                lines.append(line.content)
    return "\n".join(lines).strip()

def ocr_extract_many(images_bytes_list) -> str:
    chunks = []
    for idx, b in enumerate(images_bytes_list, start=1):
        text = ocr_extract(b)
        if text:
            chunks.append(f"[[PAGE {idx}]]\n{text}")
    return "\n\n".join(chunks).strip()

def gpt_chat(messages, temperature=0):
    # Try modern; fallback if needed
    if hasattr(gpt_client, "chat") and hasattr(gpt_client.chat, "completions"):
        resp = gpt_client.chat.completions.create(model=GPT_DEPLOYMENT, temperature=temperature, messages=messages)
    else:
        resp = gpt_client.chat_completions.create(model=GPT_DEPLOYMENT, temperature=temperature, messages=messages)
    return resp.choices[0].message.content

def gpt_ocr_text_to_questions(raw_text: str, quality_subject_ctx: str) -> dict:
    prompt = enrich_with_quality_subject(SYSTEM_PROMPT_OCR_TO_QA_BASE, **quality_subject_ctx)
    content = gpt_chat([
        {"role": "system", "content": prompt},
        {"role": "user", "content": raw_text}
    ], temperature=0)
    content = clean_model_json(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def gpt_notes_to_questions(notes_text: str, quality_subject_ctx: str) -> dict:
    prompt = enrich_with_quality_subject(SYSTEM_PROMPT_NOTES_TO_QA_BASE, **quality_subject_ctx)
    content = gpt_chat([
        {"role": "system", "content": prompt},
        {"role": "user", "content": notes_text}
    ], temperature=0)
    content = clean_model_json(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def gpt_questions_to_placeholders(questions_data: dict) -> dict:
    q = questions_data.get("questions", [])
    if len(q) > 5:
        questions_data = {"questions": q[:5]}
    content = gpt_chat([
        {"role": "system", "content": SYSTEM_PROMPT_QA_TO_PLACEHOLDERS_BASE},
        {"role": "user", "content": json.dumps(questions_data, ensure_ascii=False)}
    ], temperature=0)
    content = clean_model_json(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


# ---------------------------
# SSML + TTS helpers
# ---------------------------
def build_ssml(text: str, lang_tag: str, voice: str, rate_pct: int, pitch_semi: int, add_break_ms: int) -> str:
    # Simple SSML; Azure supports say-as etc; keep robust.
    prosody = f'rate="{rate_pct}%" pitch="{pitch_semi}st"'
    br = f'<break time="{add_break_ms}ms"/>' if add_break_ms > 0 else ""
    # Escape basic XML entities
    def esc(s): return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<speak version="1.0" xml:lang="{lang_tag}">
  <voice name="{voice}">
    <prosody {prosody}>{esc(text)}{br}</prosody>
  </voice>
</speak>"""

def tts_synthesize_and_upload(ssml: str, filename_base: str):
    """Return (audio_s3_key, audio_cdn_url) or (None, None) if disabled/unavailable."""
    if not (HAS_SPEECH and AZURE_SPEECH_KEY and AZURE_SPEECH_REGION and ssml.strip()):
        return None, None
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        # 48kHz MP3 (compatible)
        speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            return None, None
        audio_bytes = result.audio_data
        # Build S3 key
        fname = f"{filename_base}.mp3"
        s3_key = f"{AUDIO_S3_PREFIX.strip('/')}/{fname}" if AUDIO_S3_PREFIX else fname
        s3 = get_s3_client()
        s3.put_object(
            Bucket=AWS_BUCKET,
            Key=s3_key,
            Body=audio_bytes,
            ContentType="audio/mpeg",
            CacheControl="public, max-age=86400",
            ContentDisposition=f'inline; filename="{fname}"',
        )
        cdn_url = f"{CDN_MEDIA_BASE.rstrip('/')}/{s3_key}"
        return s3_key, cdn_url
    except Exception:
        return None, None


# ---------------------------
# Template merge + S3 upload + publisher meta
# ---------------------------
def build_attr_value(key: str, val: str) -> str:
    if not key.endswith("attr") or not val:
        return ""
    m = re.match(r"s(\d+)option(\d)attr$", key)
    if m and val.strip().lower() == "correct":
        return f"option-{m.group(2)}-correct"
    return val

def fill_template(template: str, data: dict) -> str:
    rendered = {}
    for k, v in data.items():
        if k.endswith("attr"):
            rendered[k] = build_attr_value(k, str(v))
        else:
            rendered[k] = "" if v is None else str(v)
    html = template
    for k, v in rendered.items():
        html = html.replace(f"{{{{{k}}}}}", v)
        html = html.replace(f"{{{{{k}|safe}}}}", v)
    return html

def inject_publisher_meta(html: str, *, site_name: str, canonical_url: str, publisher_name: str,
                          publisher_logo: str, author_name: str) -> str:
    json_ld = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": site_name or "Web Story",
        "author": {"@type": "Person", "name": author_name} if author_name else None,
        "publisher": {
            "@type": "Organization",
            "name": publisher_name or site_name,
            "logo": {"@type": "ImageObject", "url": publisher_logo} if publisher_logo else None
        },
        "mainEntityOfPage": canonical_url,
        "url": canonical_url
    }
    # prune None
    def prune(d):
        if isinstance(d, dict):
            return {k: prune(v) for k, v in d.items() if v is not None}
        return d
    json_ld = prune(json_ld)
    head_inject = f"""
<link rel="canonical" href="{canonical_url}"/>
<meta property="og:url" content="{canonical_url}"/>
<meta property="og:site_name" content="{site_name}"/>
<script type="application/ld+json">{json.dumps(json_ld, ensure_ascii=False)}</script>
"""
    # Insert before </head> if possible
    if "</head>" in html:
        return html.replace("</head>", head_inject + "\n</head>")
    return head_inject + html

def upload_html_to_s3(html_text: str, filename: str, prefix_override: str = None):
    if not filename.lower().endswith(".html"):
        filename = f"{filename}.html"

    # Use override if given, else fall back to HTML_S3_PREFIX
    effective_prefix = prefix_override if prefix_override is not None else HTML_S3_PREFIX
    s3_key = f"{effective_prefix.strip('/')}/{filename}" if effective_prefix else filename

    s3 = get_s3_client()
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=s3_key,
        Body=html_text.encode("utf-8"),
        ContentType="text/html; charset=utf-8",
        CacheControl="public, max-age=300",
        ContentDisposition=f'inline; filename="{filename}"',
    )
    cdn_url = f"{CDN_HTML_BASE.rstrip('/')}/{s3_key}"
    return s3_key, cdn_url


# ---------------------------
# 🧩 Builder UI
# ---------------------------
tab_all, = st.tabs(["All-in-one Builder"])

with tab_all:
    st.subheader("Build final AMP HTML from image(s) or structured JSON")

    # ---------- Quality tuning + Subject depth controls ----------
    with st.expander("🎛️ Content Quality & Subject Depth", expanded=True):
        col1, col2, col3 = st.columns(3)
        subject = col1.selectbox("Subject", ["General", "Physics", "Chemistry", "Biology", "Mathematics", "History", "Geography", "Civics", "Computer Science", "Economics", "English", "Hindi"], index=0)
        grade = col2.selectbox("Grade/Level", ["Generic", "K-8", "9-10", "11-12", "Undergrad", "Professional"], index=0)
        depth_level = col3.slider("Depth level", 1, 5, 3)

        subtopic = st.text_input("Subtopic (optional)", value="")

        c1, c2, c3, c4 = st.columns(4)
        difficulty = c1.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        bloom = c2.selectbox("Bloom Emphasis", ["Remember", "Understand", "Apply", "Analyze"], index=1)
        distractor_rigor = c3.selectbox("Distractor Rigor", ["Reasonable", "High (very plausible)"], index=1)
        explanation_style = c4.selectbox("Explanation Style", ["Concise (1-2 lines)", "Detailed (3-5 lines)"], index=0)
        examples_pref = st.selectbox("Examples Preference", ["No examples", "Add small real-world example if helpful"], index=1)

    # ---------- Narration + SSML / TTS ----------
    with st.expander("🗣️ Narration & SSML / TTS", expanded=True):
        add_narration = st.toggle("Generate narration text for slides", value=True)
        add_ssml = st.toggle("Generate SSML for intro + each question", value=True)
        gen_tts = st.toggle("Synthesize audio with Azure Speech (MP3) & upload to S3", value=False,
                            help="Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION. SSML will still be generated even if disabled.")

        colv1, colv2, colv3 = st.columns(3)
        default_lang = "en-US"
        voice_en = colv1.text_input("English voice", "en-US-AriaNeural")
        voice_hi = colv2.text_input("Hindi voice", "hi-IN-SwaraNeural")
        ssml_rate = colv3.slider("SSML rate (%)", 60, 140, 100)
        ssml_pitch = st.slider("SSML pitch (semitones)", -6, 6, 0)
        ssml_break = st.slider("Trailing pause (ms)", 0, 600, 200)

    # ---------- Template + Publisher integration ----------
    with st.expander("🧩 Template & Publisher Integration", expanded=True):
        up_tpl = st.file_uploader("📎 Upload AMP HTML template (.html)", type=["html", "htm"], key="tpl")
        # S3 prefix override + filename root toggle
        use_root = st.checkbox("Save HTML at bucket root (no prefix)", value=(HTML_S3_PREFIX.strip("/") == ""))
        html_prefix_override = "" if use_root else st.text_input("HTML S3 prefix", value=HTML_S3_PREFIX)
        if use_root and HTML_S3_PREFIX:
            st.caption("Will temporarily override HTML_S3_PREFIX to ''.")
        # Publisher meta
        inject_pub = st.toggle("Inject canonical + JSON-LD publisher metadata", value=True)
        site_name = st.text_input("Site name", "Suvichaar Stories")
        canonical_base = st.text_input("Canonical base (no trailing slash)", "https://stories.suvichaar.org")
        publisher_name = st.text_input("Publisher name", "Suvichaar")
        publisher_logo = st.text_input("Publisher logo URL", "")
        author_name = st.text_input("Author (optional)", "Suvichaar Team")

        # Template validator
        if up_tpl is not None:
            check = st.button("🔎 Validate template placeholders")
            if check:
                tpl_html_preview = up_tpl.getvalue().decode("utf-8", errors="replace")
                missing, optional_present = validate_template_placeholders(tpl_html_preview)
                if missing:
                    st.error("Missing required placeholders:\n" + ", ".join(missing))
                else:
                    st.success("All required placeholders found.")
                if optional_present:
                    st.info("Optional audio/SSML placeholders present: " + ", ".join(optional_present))
                else:
                    st.caption("No optional audio/SSML placeholders found. You can still generate audio; just add placeholders like s2audio_url, s2ssml, etc., to your template if you want them wired in.")

    # ---------- Input mode ----------
    st.subheader("Input")
    st.caption("Pick an input source; toggle debug to preview OCR/JSON.")
    mode = st.radio(
        "Choose input",
        ["Notes image(s) (OCR → generate quiz JSON)", "Quiz image (OCR → parse existing MCQs)", "Structured JSON (skip OCR)"],
        horizontal=False,
    )
    show_debug = st.toggle("Show OCR / JSON previews", value=False)

    questions_data = None

    quality_subject_ctx = dict(
        subject=subject,
        grade=grade,
        subtopic=subtopic,
        depth_level=depth_level,
        difficulty=difficulty,
        bloom=bloom,
        distractor_rigor=distractor_rigor,
        explanation_style=explanation_style,
        examples_pref=examples_pref,
    )

    if mode == "Notes image(s) (OCR → generate quiz JSON)":
        up_imgs = st.file_uploader(
            "📎 Upload notes image(s) (JPG/PNG/WebP/TIFF) — multiple allowed",
            type=["jpg", "jpeg", "png", "webp", "tiff"],
            accept_multiple_files=True,
            key="notes_imgs"
        )
        if up_imgs:
            if show_debug:
                for i, f in enumerate(up_imgs, start=1):
                    try:
                        st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                                 caption=f"Notes page {i}", use_container_width=True)
                    except Exception:
                        pass
            try:
                with st.spinner("🔍 OCR (Azure Document Intelligence) on all pages…"):
                    all_bytes = [f.getvalue() for f in up_imgs]
                    notes_text = ocr_extract_many(all_bytes)
                if not notes_text.strip():
                    st.error("OCR returned empty text. Try clearer images.")
                    st.stop()
                if show_debug:
                    with st.expander("📄 OCR Notes Text"):
                        st.text(notes_text[:8000] if len(notes_text) > 8000 else notes_text)

                with st.spinner("📝 Generating 5 MCQs from notes…"):
                    questions_data = gpt_notes_to_questions(notes_text, quality_subject_ctx)
                if show_debug and questions_data:
                    with st.expander("🧱 Generated Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:8000], language="json")
            except Exception as e:
                st.error(f"Failed to process notes → quiz JSON: {e}")
                st.stop()

    elif mode == "Quiz image (OCR → parse existing MCQs)":
        up_img = st.file_uploader("📎 Upload quiz image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="quiz_img")
        if up_img:
            img_bytes = up_img.getvalue()
            try:
                if show_debug:
                    st.image(Image.open(io.BytesIO(img_bytes)).convert("RGB"), caption="Uploaded quiz image", use_container_width=True)
                with st.spinner("🔍 OCR (Azure Document Intelligence)…"):
                    raw_text = ocr_extract(img_bytes)
                if not raw_text.strip():
                    st.error("OCR returned empty text. Try a clearer image.")
                    st.stop()
                if show_debug:
                    with st.expander("📄 OCR Text"):
                        st.text(raw_text[:8000] if len(raw_text) > 8000 else raw_text)
                with st.spinner("🤖 Parsing OCR into questions JSON…"):
                    questions_data = gpt_ocr_text_to_questions(raw_text, quality_subject_ctx)
                if show_debug and questions_data:
                    with st.expander("🧱 Structured Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:8000], language="json")
            except Exception as e:
                st.error(f"Failed to process image → JSON: {e}")
                st.stop()

    else:  # Structured JSON
        up_json = st.file_uploader("📎 Upload structured questions JSON", type=["json"], key="json")
        if up_json:
            try:
                questions_data = json.loads(up_json.getvalue().decode("utf-8"))
                if show_debug:
                    with st.expander("🧱 Structured Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:8000], language="json")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

    # Build button
    # Allow building only if both data + template are provided
    build = st.button("🛠️ Build final HTML", disabled=not ((questions_data is not None) and (up_tpl is not None)))

    if build and questions_data and up_tpl:
        try:
            # → placeholders
            with st.spinner("🧩 Generating placeholders…"):
                placeholders = gpt_questions_to_placeholders(questions_data)
                if show_debug:
                    with st.expander("🧩 Placeholder JSON"):
                        st.code(json.dumps(placeholders, ensure_ascii=False, indent=2)[:12000], language="json")

            # Derive language for SSML/voices
            sample_text = (placeholders.get("s2question1") or placeholders.get("s1text1") or "")[:180]
            lang_tag = detect_lang(sample_text)
            voice = voice_hi if lang_tag.startswith("hi") else voice_en

            # Build intro & slide SSML if requested
            if add_ssml:
                # Intro
                if placeholders.get("s1narration", ""):
                    placeholders["s1ssml"] = build_ssml(placeholders["s1narration"], lang_tag, voice, ssml_rate, ssml_pitch, ssml_break)
                # Slides
                for n in range(2, 7):
                    key = f"s{n}narration"
                    ssml_key = f"s{n}ssml"
                    if placeholders.get(key, ""):
                        placeholders[ssml_key] = build_ssml(placeholders[key], lang_tag, voice, ssml_rate, ssml_pitch, ssml_break)

            # Optional TTS synthesis & upload
            # Will no-op if gen_tts false or Speech not configured
            if gen_tts and add_ssml:
                tsstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_slug = slugify(placeholders.get("storytitle") or placeholders.get("pagetitle") or "story")
                # Intro
                if placeholders.get("s1ssml"):
                    k, url = tts_synthesize_and_upload(placeholders["s1ssml"], f"{base_slug}_{tsstamp}_intro")
                    if url:
                        placeholders["s1audio_url"] = url
                # Slides
                for n in range(2, 7):
                    ssml_text = placeholders.get(f"s{n}ssml")
                    if ssml_text:
                        k, url = tts_synthesize_and_upload(ssml_text, f"{base_slug}_{tsstamp}_s{n}")
                        if url:
                            placeholders[f"s{n}audio_url"] = url

            # Read template
            template_html = up_tpl.getvalue().decode("utf-8", errors="replace")

            # Merge placeholders
            final_html = fill_template(template_html, placeholders)

            # Publisher integration (canonical + JSON-LD)
            if inject_pub:
                # Build canonical using slug + timestamp and prefix/root choice
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                slug = slugify(placeholders.get("storytitle") or placeholders.get("pagetitle") or "webstory")
                # Honor root preference
                html_prefix_runtime = "" if use_root else html_prefix_override
                path_part = f"{slug}_{ts}.html"
                canonical_path = f"{path_part}" if not html_prefix_runtime else f"{html_prefix_runtime.strip('/')}/{path_part}"
                canonical_url = f"{canonical_base.rstrip('/')}/{canonical_path}"
                final_html = inject_publisher_meta(
                    final_html,
                    site_name=site_name,
                    canonical_url=canonical_url,
                    publisher_name=publisher_name,
                    publisher_logo=publisher_logo,
                    author_name=author_name,
                )
            else:
                # Keep as-is if no injection
                canonical_url = None

            # Compute filename: use slug + timestamp (root or prefix controlled at upload)
            ts_name = f"{slugify(placeholders.get('storytitle') or placeholders.get('pagetitle') or 'final_quiz')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path(ts_name).write_text(final_html, encoding="utf-8")

            # Upload with runtime prefix override
            prev_prefix = HTML_S3_PREFIX
            HTML_S3_PREFIX = "" if use_root else html_prefix_override

            with st.spinner("☁️ Uploading HTML to S3…"):
                s3_key, cdn_url = upload_html_to_s3(final_html, ts_name)

            # restore global in case of repeated builds
            HTML_S3_PREFIX = prev_prefix

            st.success(f"✅ Final HTML generated and uploaded to S3: s3://{AWS_BUCKET}/{s3_key}")
            st.markdown(f"**CDN URL:** {cdn_url}")

            if canonical_url:
                st.caption(f"Canonical: {canonical_url}")

            with st.expander("🔍 HTML Preview (source)"):
                st.code(final_html[:120000], language="html")

            st.download_button(
                "⬇️ Download final HTML",
                data=final_html.encode("utf-8"),
                file_name=ts_name,
                mime="text/html"
            )

            # Live viewer
            st.markdown("### 👀 Live HTML Preview")
            h = st.slider("Preview height (px)", 400, 1600, 900, 50)
            full_width = st.checkbox("Force full viewport width (100vw)", value=True)
            style = f"width: {'100vw' if full_width else '100%'}; height: {h}px; border: 0; margin: 0; padding: 0;"
            st_html(final_html, height=h, scrolling=True) if not full_width else st_html(
                f'<div style="position:relative;left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;{style}">{final_html}</div>',
                height=h,
                scrolling=True
            )

            # Hints
            if gen_tts and not HAS_SPEECH:
                st.warning("Audio synthesis requested but Azure Speech SDK isn’t installed/available. SSML fields were generated, but no MP3s were created.")
            if gen_tts and HAS_SPEECH and not (AZURE_SPEECH_KEY and AZURE_SPEECH_REGION):
                st.warning("Azure Speech credentials missing. SSML generated; audio not synthesized.")

        except Exception as e:
            st.error(f"Build failed: {e}")
    else:
        st.info("Upload an input (notes/quiz image(s) or JSON) **and** a template to enable the Build button.")
