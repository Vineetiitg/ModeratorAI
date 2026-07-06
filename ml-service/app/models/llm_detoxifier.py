"""
SafeChat — LLM-Powered Text Detoxifier (Gemini API)

Intent-preserving style transfer for toxic messages.
Uses Google Gemini to rewrite toxic text while:
  - Preserving the speaker's original meaning and intent
  - Matching the detected language (Hindi, Hinglish, English, etc.)
  - Maintaining conversational tone
  - Removing toxicity

Supports async streaming for real-time display via WebSocket.
Falls back to language-specific templates if the API is unavailable.
"""

from typing import AsyncGenerator, Dict, List, Optional
import anyio

from loguru import logger

from app.config import settings
from app.utils.preprocessing import detect_language


def _is_devanagari_script(text: str) -> bool:
    """Check if the text contains significant Devanagari script (>30% of alphabetic chars)."""
    devanagari_chars = 0
    total_alpha = 0
    for char in text:
        if char.isalpha():
            total_alpha += 1
            if '\u0900' <= char <= '\u097F':
                devanagari_chars += 1
    if total_alpha == 0:
        return False
    return (devanagari_chars / total_alpha) > 0.3


# ── Multilingual Fallback Templates ────────────────────────────────────
# Used when Gemini API is unavailable or fails.

FALLBACK_TEMPLATES = {
    "en": "Let's keep the conversation respectful.",
    "hi": "कृपया बातचीत को सम्मानजनक बनाए रखें।",
    "hi-en": "Yaar, thoda respectfully baat karte hain.",
    "bn": "দয়া করে কথোপকথনটি সম্মানজনক রাখুন।",
    "ta": "தயவுசெய்து உரையாடலை மரியாதையாக வைத்திருங்கள்.",
    "te": "దయచేసి సంభాషణను గౌరవంగా ఉంచండి.",
    "kn": "ದಯವಿಟ್ಟು ಸಂಭಾಷಣೆಯನ್ನು ಗೌರವಯುತವಾಗಿ ಇರಿಸಿ.",
    "ml": "ദയവായി സംഭാഷണം മാന്യമായി നിലനിർത്തുക.",
    "gu": "કૃપા કરી વાતચીતને સન્માનજનક રાખો.",
    "pa": "ਕਿਰਪਾ ਕਰਕੇ ਗੱਲਬਾਤ ਨੂੰ ਸਤਿਕਾਰਯੋਗ ਰੱਖੋ.",
    "indic-en": "Let's keep the conversation respectful, please.",
}

# ── Language Display Names ─────────────────────────────────────────────

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (Devanagari)",
    "hi-en": "Hinglish (Hindi-English code-mixed)",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "indic-en": "Indian language mixed with English",
}


def _build_detox_prompt(
    text: str,
    language: str,
    categories: Optional[Dict[str, float]] = None,
    context: Optional[List[str]] = None,
) -> str:
    """
    Build a structured prompt for intent-preserving detoxification.

    The prompt includes:
      - Toxicity categories and scores (so the LLM knows what to fix)
      - Detected language (so the LLM responds in the same language)
      - Conversation context (so the LLM preserves conversational flow)
      - Explicit rules for style transfer
    """
    lang_name = LANGUAGE_NAMES.get(language, language)

    # Format toxicity categories
    categories_str = "Unknown"
    if categories:
        flagged = {k: v for k, v in categories.items() if v >= settings.THRESHOLD_SAFE}
        if flagged:
            categories_str = ", ".join(f"{k}: {v:.2f}" for k, v in sorted(flagged.items(), key=lambda x: -x[1]))

    # Format conversation context
    context_block = ""
    if context and len(context) > 0:
        turns = context[-4:]  # Last 4 messages
        context_lines = "\n".join(f"  [{i+1}] {turn}" for i, turn in enumerate(turns))
        context_block = f"""
Conversation context (previous messages):
{context_lines}
"""

    return f"""You are a chat message rewriter for a content safety system.

TASK: Rewrite the following toxic message to remove toxicity while preserving the speaker's original intent and meaning.

TOXIC MESSAGE: "{text}"

Detected toxicity: {categories_str}
Detected language: {lang_name}
{context_block}
RULES:
1. Respond ONLY with the rewritten message — no explanations, no quotes, no preamble.
2. Write in the SAME language as the input. If the input is Hinglish (Hindi words in Latin script mixed with English), respond in Hinglish.
3. Keep the conversational tone — don't make it sound formal or corporate.
4. Preserve what the person was trying to communicate, just remove the abusive/toxic parts.
5. Keep it concise — similar length to the original message.
6. If the message is a greeting, question, or request buried under insults, extract and preserve that core intent.

REWRITTEN MESSAGE:"""


def _is_generic_scolding(text: str) -> bool:
    """Check if generated text is just a polite scolding/reminder instead of an intent-preserving rewrite."""
    if not text:
        return True
    scolding_words = [
        "kripya", "respectful", "vinaamrata", "sabhyata", "apashabd", "bhasha",
        "polite", "language", "maintain", "baat karte hain", "samvad",
        "opinion respectfully", "avoid using offensive", "offensive language"
    ]
    return any(w.lower() in text.lower() for w in scolding_words)


def _clean_preserve_intent(text: str, lang: str) -> str:
    """
    Intelligently scrubs swear words, insults, slurs, and threats from the user's
    exact message while preserving the core conversational intent and structure.
    """
    import re
    cleaned = text

    # Hinglish & Hindi Romanized toxic words to remove or replace
    hinglish_map = {
        r'\b(bewakoof|gadhha|gadhe|chutiya|chomu|ullu|lukkha|nalayak)\b': 'naasamajh',
        r'\b(saale|saala|kamine|kamini|harami|haramkhor|dalla|suar|kutta|kutte|bhikari)\b': '',
        r'\b(bhenchod|madarchod|machod|boshdike|bsdk|mc|bc)\b': 'bhai',
        r'\b(bakwas mat kar|bakwas band kar|faltu baat band kar)\b': 'sahi se baat karo',
        r'\b(muh tod duga|maar duga|jawani nikal dunga|aukat me reh|aukat dekh|aukat)\b': 'shanti aur hadd me raho',
        r'\b(nikal yahan se|nikal pehli fursat mein)\b': 'kripya abhi yahan se jao',
    }

    # Devanagari Hindi toxic words to remove or replace
    hindi_map = {
        r'बेवकूफ|गधा|चूतिया|हरामी|नालायक|कमीने|सूअर|कुत्ता': 'नासमझ',
        r'साले|साला|हरामखोर|भिखारी': '',
        r'बहनचोद|मादरचोद': 'भाई',
        r'बकवास मत कर|बकवास बंद कर': 'सही से बात करो',
        r'मुंह तोड़ दूंगा|मार दूंगा|औकात में रह': 'शांति से बात करो',
    }

    # English toxic words to remove or replace
    english_map = {
        r'\b(idiot|moron|retard|retarded|fool|clown|freak|loser|parasite|scum|trash|bastard)\b': 'mistaken',
        r'\b(shut up)\b': 'please stop talking',
        r'\b(go to hell)\b': 'please leave this conversation',
        r'\b(fucking|fuck|bitch|shit|asshole|damn)\b': '',
    }

    # Apply regex replacements based on script/language
    for pattern, replacement in {**hinglish_map, **hindi_map, **english_map}.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Clean up extra whitespace and commas
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r' ,\s*', ', ', cleaned)
    cleaned = re.sub(r'^\s*,\s*', '', cleaned)

    # Capitalize first letter
    if cleaned and len(cleaned) > 0:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned if cleaned and len(cleaned) > 2 else text


class LLMDetoxifier:
    """
    LLM-powered detoxification via Google Gemini API (or AICredits / proxy platforms).

    Features:
      - Intent-preserving style transfer
      - Language-matched output (responds in same language as input)
      - Async streaming for real-time display
      - Graceful fallback to intent-preserving local scrubber
    """

    def __init__(self):
        self._client = None
        self._available = False
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Gemini / AICredits API client."""
        if not settings.GEMINI_API_KEY:
            logger.warning(
                "SAFECHAT_GEMINI_API_KEY not set. "
                "LLM detoxification will use fallback templates only."
            )
            return

        try:
            if settings.GEMINI_API_KEY.startswith("sk-") or getattr(settings, "GEMINI_BASE_URL", None):
                from openai import AsyncOpenAI
                base_url = getattr(settings, "GEMINI_BASE_URL", "https://api.aicredits.in/v1")
                self._client_type = "openai_compatible"
                self._client = AsyncOpenAI(api_key=settings.GEMINI_API_KEY, base_url=base_url)
                self._available = True
                logger.success(f"AICredits / OpenAI-compatible client initialized (model: {settings.GEMINI_MODEL}, base_url: {base_url})")
            else:
                from google import genai
                self._client_type = "google_genai"
                self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
                self._available = True
                logger.success(f"Google Gemini client initialized (model: {settings.GEMINI_MODEL})")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")

    @property
    def is_available(self) -> bool:
        """Whether LLM detoxification is available (API key set & client initialized)."""
        return self._available

    async def detoxify(
        self,
        text: str,
        toxicity_categories: Optional[Dict[str, float]] = None,
        target_language: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> Dict:
        """
        Detoxify a message using Gemini API (high nuance, intent-preserving style transfer).
          - Route 1: Route directly to Gemini API
          - Fallback: Intent-preserving local scrubber
        """
        lang = target_language or detect_language(text)

        # ── Route 1: Gemini API
        if self._available:
            try:
                result = await self._llm_detoxify(text, lang, toxicity_categories, context)
                if result and not _is_generic_scolding(result):
                    return {
                        "original": text,
                        "detoxified": result,
                        "method": "gemini",
                        "language": lang,
                        "confidence": 0.95,
                    }
            except Exception as e:
                logger.error(f"Gemini detoxification failed: {e}")

        # ── Route 2: Intent-Preserving Local Scrubber (Fallback)
        cleaned_text = _clean_preserve_intent(text, lang)
        return {
            "original": text,
            "detoxified": cleaned_text,
            "method": "intent_preserving_scrubber",
            "language": lang,
            "confidence": 0.85,
        }

    async def _llm_detoxify(
        self,
        text: str,
        lang: str,
        categories: Optional[Dict[str, float]] = None,
        context: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Call Gemini API for detoxification (non-streaming)."""
        if not self._client:
            return None

        prompt = _build_detox_prompt(text, lang, categories, context)

        try:
            if getattr(self, "_client_type", None) == "openai_compatible":
                response = await self._client.chat.completions.create(
                    model=settings.GEMINI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150,
                )
                result = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else None
            else:
                response = await self._client.aio.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                )
                result = response.text.strip() if response.text else None

            # Strip surrounding quotes if the LLM added them
            if result and len(result) >= 2:
                if (result[0] == '"' and result[-1] == '"') or (result[0] == "'" and result[-1] == "'"):
                    result = result[1:-1].strip()

            return result if result else None

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    async def detoxify_stream(
        self,
        text: str,
        toxicity_categories: Optional[Dict[str, float]] = None,
        target_language: Optional[str] = None,
        context: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream detoxified tokens for real-time display via WebSocket using Gemini API.
        """
        lang = target_language or detect_language(text)

        if self._available and self._client:
            prompt = _build_detox_prompt(text, lang, toxicity_categories, context)

            try:
                if getattr(self, "_client_type", None) == "openai_compatible":
                    stream = await self._client.chat.completions.create(
                        model=settings.GEMINI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=150,
                        stream=True,
                    )
                    full_generated = ""
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_generated += content
                            yield content
                    if not _is_generic_scolding(full_generated):
                        return
                else:
                    response = await self._client.aio.models.generate_content_stream(
                        model=settings.GEMINI_MODEL,
                        contents=prompt,
                    )

                    full_generated = ""
                    async for chunk in response:
                        if chunk.text:
                            full_generated += chunk.text
                            yield chunk.text
                    if not _is_generic_scolding(full_generated):
                        return
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}")

        # Fallback: yield intent-preserving cleaned text as a single chunk
        cleaned_text = _clean_preserve_intent(text, lang)
        words = cleaned_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    def get_info(self) -> Dict:
        """Return detoxifier metadata for health checks."""
        return {
            "mode": "gemini" if self._available else "fallback",
            "gemini_api_configured": bool(settings.GEMINI_API_KEY),
        }
