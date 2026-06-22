"""
SafeChat — Real-Time Toxic Chat Moderation Dashboard
=====================================================
A comprehensive Streamlit frontend that showcases:
  • Live chat moderation with per-category toxicity probabilities
  • Gemini 2.0 Flash detoxified alternative suggestions
  • Severity badge system (SAFE / LOW / MEDIUM / HIGH)
  • Conversation context awareness
  • Continuous learning feedback loop
  • Session analytics sidebar

Run:  streamlit run app.py
"""

import streamlit as st
import uuid
import time
import json
from datetime import datetime
from api_client import check_health, moderate, detoxify, get_feedback_stats, submit_feedback

# ─── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="SafeChat · Real-Time Moderation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium Dark-Mode Glassmorphism CSS ──────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Global ─── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%) !important;
    color: #e0e0e0;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #a78bfa !important;
}

/* ─── Glass Cards ─── */
.glass {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(167, 139, 250, 0.08);
}

/* ─── Severity Badges ─── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-safe    { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-low     { background: rgba(234,179,8,0.15);  color: #eab308; border: 1px solid rgba(234,179,8,0.3);  }
.badge-medium  { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.badge-high    { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3);  }

/* ─── Progress Bars ─── */
.prob-bar-container {
    display: flex;
    align-items: center;
    margin: 3px 0;
    gap: 8px;
}
.prob-label {
    min-width: 100px;
    font-size: 0.78rem;
    color: #9ca3af;
    text-align: right;
}
.prob-track {
    flex: 1;
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 5px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.prob-value {
    min-width: 48px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #d1d5db;
}

/* ─── Toxic Message Card ─── */
.toxic-card {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
}

/* ─── Suggestion Card ─── */
.suggestion-card {
    background: rgba(34, 197, 94, 0.06);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 12px;
    padding: 16px;
    margin-top: 12px;
}
.suggestion-card strong { color: #22c55e; }

/* ─── Safe Message Card ─── */
.safe-card {
    background: rgba(34, 197, 94, 0.04);
    border: 1px solid rgba(34, 197, 94, 0.12);
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
}

/* ─── Metric Tiles ─── */
.metric-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.metric-tile .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-tile .label {
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ─── Header gradient text ─── */
.gradient-text {
    background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* ─── Chat input ─── */
.stChatInput > div {
    border-radius: 12px !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stChatInput input {
    color: #e0e0e0 !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 3px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "total_safe" not in st.session_state:
    st.session_state.total_safe = 0
if "total_toxic" not in st.session_state:
    st.session_state.total_toxic = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0


# ─── Helper Functions ─────────────────────────────────────────────────────────

def severity_badge(severity: str) -> str:
    cls = f"badge-{severity.lower()}"
    return f'<span class="badge {cls}">{severity}</span>'


def prob_bar(label: str, value: float) -> str:
    """Render a single category probability bar."""
    pct = value * 100
    if pct > 75:
        color = "#ef4444"
    elif pct > 50:
        color = "#f97316"
    elif pct > 30:
        color = "#eab308"
    else:
        color = "#22c55e"
    return f"""
    <div class="prob-bar-container">
        <div class="prob-label">{label}</div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-value">{pct:.1f}%</div>
    </div>
    """


def render_categories(categories: dict) -> str:
    """Render all 6 category probability bars."""
    nice_names = {
        "toxic": "Toxic",
        "severe_toxic": "Severe Toxic",
        "obscene": "Obscene",
        "identity_hate": "Identity Hate",
        "insult": "Insult",
        "threat": "Threat",
    }
    bars = ""
    for key in ["toxic", "severe_toxic", "obscene", "insult", "identity_hate", "threat"]:
        val = categories.get(key, 0.0)
        bars += prob_bar(nice_names.get(key, key), val)
    return bars


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<h2 class="gradient-text">🛡️ SafeChat</h2>', unsafe_allow_html=True)
    st.caption("Context-Aware Multilingual Content Safety Platform")
    st.divider()

    # Health check
    health = check_health()
    if health:
        st.success("ML Service Online", icon="✅")
    else:
        st.error("ML Service Offline", icon="🔴")
        st.caption("Start the ML service on port 8001")

    st.divider()

    # Session analytics
    st.markdown("### 📊 Session Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="value" style="color:#22c55e">{st.session_state.total_safe}</div>
            <div class="label">Safe</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="value" style="color:#ef4444">{st.session_state.total_toxic}</div>
            <div class="label">Toxic</div>
        </div>
        """, unsafe_allow_html=True)

    total = st.session_state.total_safe + st.session_state.total_toxic
    if total > 0:
        safe_pct = (st.session_state.total_safe / total) * 100
        st.markdown(f"""
        <div class="metric-tile" style="margin-top:10px">
            <div class="value" style="color:#818cf8">{safe_pct:.0f}%</div>
            <div class="label">Safety Rate</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.total_time > 0 and total > 0:
        avg_ms = st.session_state.total_time / total
        st.markdown(f"""
        <div class="metric-tile" style="margin-top:10px">
            <div class="value" style="color:#f97316">{avg_ms:.0f}ms</div>
            <div class="label">Avg Latency</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Model info
    st.markdown("### 🧠 Model Stack")
    st.markdown("""
    - **Classifier:** HingBERT (fine-tuned)
    - **Gatekeeper:** HingBERT / TextDetox
    - **Detoxifier:** Gemini 2.0 Flash
    - **Categories:** 6 multi-label
    """)

    st.divider()

    # Quick test messages
    st.markdown("### 🧪 Quick Test Messages")
    test_messages = {
        "🟢 Safe English": "Hey, how are you doing today?",
        "🟢 Safe Hindi": "नमस्ते भाई, कैसे हो?",
        "🟢 Safe Hinglish": "Bhai party kab de raha hai tu?",
        "🔴 Toxic English": "You're such an idiot, shut up!",
        "🔴 Toxic Hindi": "तू बहुत बड़ा बेवकूफ है, चुप रह साले",
        "🔴 Toxic Hinglish": "bhenchod bakwas mat kar harami",
    }

    for label, msg in test_messages.items():
        if st.button(label, key=f"test_{label}", use_container_width=True):
            st.session_state["_inject_msg"] = msg

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_safe = 0
        st.session_state.total_toxic = 0
        st.session_state.total_time = 0
        st.rerun()


# ─── Main Content ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <h1 class="gradient-text" style="font-size:2.2rem; margin-bottom:4px;">
        🛡️ SafeChat · Real-Time Moderation
    </h1>
    <p style="color:#6b7280; font-size:0.9rem;">
        Context-aware multilingual toxicity detection with intent-preserving style transfer
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Render Chat History ──────────────────────────────────────────────────────

for idx, msg in enumerate(st.session_state.messages):
    mod = msg.get("moderation")

    if mod and mod.get("is_toxic"):
        # ── TOXIC MESSAGE ──
        flagged = [k for k, v in mod.get("categories", {}).items() if v > 0.3]
        flagged_str = ", ".join(flagged) if flagged else "general"

        st.markdown(f"""
        <div class="toxic-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div>
                    <span style="font-size:0.75rem; color:#6b7280;">
                        {msg.get('timestamp', '')} · {mod.get('detected_language', '?')} · {mod.get('inference_time_ms', '?')}ms
                    </span>
                </div>
                <div>
                    {severity_badge(mod.get('severity', 'HIGH'))}
                </div>
            </div>
            <div style="font-size:0.95rem; color:#fca5a5; margin-bottom:14px;">
                ⚠️ <span style="text-decoration:line-through; opacity:0.7;">{msg['content']}</span>
            </div>
            <div style="font-size:0.78rem; color:#9ca3af; margin-bottom:6px; font-weight:500;">
                CATEGORY PROBABILITIES
            </div>
            {render_categories(mod.get('categories', dict()))}
            <div style="margin-top:6px; font-size:0.8rem; color:#6b7280;">
                Overall Score: <strong style="color:#ef4444">{mod.get('overall_score', 0):.2%}</strong>
                &nbsp;·&nbsp; Flagged: <strong style="color:#fbbf24">{flagged_str}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show suggestion
        suggestion = mod.get("suggestion") or msg.get("detoxified")
        if suggestion:
            st.markdown(f"""
            <div class="suggestion-card">
                <strong>✨ Suggested Alternative (Gemini 2.0 Flash):</strong>
                <div style="margin-top:8px; font-size:0.95rem; color:#d1fae5;">
                    "{suggestion}"
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── SAFE MESSAGE ──
        lang = mod.get("detected_language", "?") if mod else "?"
        latency = mod.get("inference_time_ms", "?") if mod else "?"
        score = mod.get("overall_score", 0) if mod else 0

        st.markdown(f"""
        <div class="safe-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:0.75rem; color:#6b7280;">
                    {msg.get('timestamp', '')} · {lang} · {latency}ms
                </span>
                {severity_badge('SAFE')}
            </div>
            <div style="font-size:0.95rem; color:#d1fae5;">
                ✅ {msg['content']}
            </div>
            <div style="margin-top:8px; font-size:0.78rem; color:#4ade80;">
                Overall toxicity: {score:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Chat Input ───────────────────────────────────────────────────────────────

# Check if a test message was injected from sidebar
injected = st.session_state.pop("_inject_msg", None)
prompt = st.chat_input("Type a message in English, Hindi, or Hinglish...")
input_text = injected or prompt

if input_text:
    timestamp = datetime.now().strftime("%H:%M:%S")

    with st.spinner("🔍 Analyzing message..."):
        mod_result = moderate(input_text, context=None)

    msg_obj = {
        "id": str(uuid.uuid4()),
        "content": input_text,
        "timestamp": timestamp,
        "moderation": mod_result,
    }

    if mod_result:
        if mod_result.get("is_toxic"):
            st.session_state.total_toxic += 1
            # If suggestion wasn't provided by the /moderate endpoint, explicitly detoxify
            if not mod_result.get("suggestion"):
                with st.spinner("✨ Generating polite alternative..."):
                    detox_result = detoxify(input_text, context=context_msgs)
                    if detox_result and detox_result.get("detoxified"):
                        msg_obj["detoxified"] = detox_result["detoxified"]
        else:
            st.session_state.total_safe += 1

        st.session_state.total_time += mod_result.get("inference_time_ms", 0)
    else:
        # API unreachable — still add message
        st.session_state.total_safe += 1

    st.session_state.messages.append(msg_obj)
    st.rerun()
