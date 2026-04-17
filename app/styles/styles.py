import streamlit as st


def load_styles():
    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root palette ── */
    :root {
        --cream:   #FAF7F2;
        --ink:     #1A1208;
        --amber:   #D4873A;
        --rust:    #B84C2E;
        --sand:    #E8DDD0;
        --muted:   #7A6A58;
        --card-bg: #FFFFFF;
        --radius:  14px;
    }

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--cream) !important;
        color: var(--ink);
    }

    /* ── Hero header ── */
    .hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    width: 100%;
    display: block;
    }
    .hero-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--amber);
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 700;
        line-height: 1.1;
        color: var(--ink);
        margin: 0 0 0.6rem;
    }
    .hero-title em {
        color: var(--amber);
        font-style: italic;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--muted);
        font-weight: 300;
        max-width: 540px;
        margin: 0 auto;
        line-height: 1.6;
        text-align: center;
    }
    .divider {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2rem auto;
        max-width: 260px;
    }
    .divider-line { flex: 1; height: 1px; background: var(--sand); }
    .divider-dot  { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); }

    /* ── Section card ── */
    .section-card {
        background: var(--card-bg);
        border-radius: var(--radius);
        border: 1.5px solid var(--sand);
        padding: 1.8rem 1.6rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(26,18,8,0.05);
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title span { font-style: italic; color: var(--amber); }

    /* ── Streamlit widget overrides ── */
    .stSelectbox > label,
    .stMultiSelect > label,
    .stNumberInput > label,
    .stRadio > label,
    .stSlider > label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        margin-bottom: 0.2rem !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-color: var(--sand) !important;
        border-radius: 8px !important;
        background: var(--cream) !important;
    }
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 3px rgba(212,135,58,0.18) !important;
    }

    /* Radio buttons */
    .stRadio div[role="radiogroup"] {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .stRadio div[role="radiogroup"] label {
        background: var(--cream);
        border: 1.5px solid var(--sand);
        border-radius: 8px;
        padding: 0.38rem 1rem;
        font-size: 0.88rem !important;
        font-weight: 400 !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        cursor: pointer;
        transition: all 0.18s;
        color: var(--ink) !important;
    }
    .stRadio div[role="radiogroup"] label:hover {
        border-color: var(--amber);
        background: #FEF6EC;
    }

    /* Price range pills */
    .price-pill-row {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 0.3rem;
    }

    /* Submit button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--amber) 0%, var(--rust) 100%);
        color: #fff;
        border: none;
        border-radius: var(--radius);
        padding: 0.9rem 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        cursor: pointer;
        transition: opacity 0.2s, transform 0.15s;
    }
    .stButton > button:hover {
        opacity: 0.92;
        transform: translateY(-1px);
    }

    /* ── Result banner ── */
    .result-banner {
        background: linear-gradient(135deg, #1A1208 0%, #3D2B0E 100%);
        border-radius: var(--radius);
        padding: 2.4rem 2rem;
        text-align: center;
        color: #fff;
        margin-top: 1.4rem;
    }
    .result-label {
        font-size: 0.78rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--amber);
        margin-bottom: 0.5rem;
    }
    .result-score {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
        color: #fff;
    }
    .result-score sup { font-size: 2rem; vertical-align: super; color: var(--amber); }
    .result-stars { font-size: 1.4rem; margin: 0.5rem 0 0.3rem; }
    .result-verdict {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.7);
        font-weight: 300;
        letter-spacing: 0.04em;
    }

    /* ── Info chips ── */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.2rem;
        justify-content: center;
    }
    .chip {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 0.28rem 0.9rem;
        font-size: 0.78rem;
        color: rgba(255,255,255,0.75);
        letter-spacing: 0.04em;
    }

    /* footer note */
    .footer-note {
        text-align: center;
        font-size: 0.76rem;
        color: var(--muted);
        margin-top: 2.5rem;
        padding-bottom: 2rem;
    }

    /* hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)