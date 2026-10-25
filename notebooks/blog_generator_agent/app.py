"""
Blog Studio - Streamlit front-end for the LangGraph blog generator.

Place this file next to graph.py, agents.py and state.py
(notebooks/blog_generator_agent/app.py) and run:

    streamlit run app.py
"""
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from graph import MAX_REVISIONS, build_blog_graph

load_dotenv()

st.set_page_config(
    page_title="Blog Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────── Styling ─────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --void:   #04060C;
    --deep:   #070B16;
    --panel:  #0B1222;
    --line:   rgba(120, 165, 230, .20);
    --text:   #E8EEF8;
    --muted:  #8FA3BF;
    --blue:   #4DA3FF;
    --ice:    #9AD1FF;
    --ember:  #FF9A4D;
    --flare:  #FF6A3D;
}

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', sans-serif !important;
}

/* ---------- page: deep space ---------- */
.stApp, [data-testid="stAppViewContainer"] {
    color: var(--text);
    background:
        radial-gradient(1100px 520px at 50% -8%, rgba(60, 120, 220, .28), transparent 62%),
        radial-gradient(700px 380px at 92% 105%, rgba(255, 130, 60, .13), transparent 65%),
        radial-gradient(700px 380px at 6% 100%, rgba(60, 120, 220, .15), transparent 65%),
        var(--void) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060A14 0%, #04060C 100%) !important;
    border-right: 1px solid var(--line);
}
.block-container { padding-top: 3.5rem; padding-bottom: 4rem; max-width: 1100px; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4, label, p, li, span { color: inherit; }
hr { border-color: var(--line) !important; }

/* ---------- hero: Gargantua ---------- */
.hero {
    background:
        radial-gradient(600px 220px at 78% 118%, rgba(77,163,255,.55), transparent 70%),
        radial-gradient(420px 160px at 88% 100%, rgba(255,154,77,.45), transparent 70%),
        linear-gradient(135deg, #060B1A 0%, #0A1530 55%, #101B3D 100%);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 36px 40px;
    color: #fff;
    margin-bottom: 22px;
    box-shadow: 0 24px 60px -28px rgba(77,163,255,.55), inset 0 1px 0 rgba(255,255,255,.06);
    position: relative;
    overflow: hidden;
}
/* the bent accretion-disk arc */
.hero::after {
    content: "";
    position: absolute; right: 6%; bottom: -150px;
    width: 300px; height: 300px; border-radius: 50%;
    border: 3px solid rgba(255, 190, 120, .75);
    box-shadow:
        0 0 24px 2px rgba(255,154,77,.55),
        0 0 70px 8px rgba(77,163,255,.35),
        inset 0 0 40px rgba(77,163,255,.35);
    background: radial-gradient(circle, #000 52%, rgba(0,0,0,0) 72%);
}
.hero h1 {
    color:#fff; font-size: 2.2rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -.02em;
    text-shadow: 0 0 28px rgba(120,180,255,.45);
}
.hero p  { color: rgba(220,232,250,.82); font-size: 1.02rem; margin: 0; max-width: 52%; }

/* ---------- stepper ---------- */
.stepper { display:flex; gap:10px; margin: 6px 0 26px 0; flex-wrap: wrap; }
.step {
    flex: 1 1 150px;
    display:flex; align-items:center; gap:10px;
    padding: 12px 16px; border-radius: 14px;
    background: rgba(120,165,230,.05);
    border: 1px solid var(--line);
    color: var(--muted);
    font-weight: 600; font-size: .92rem;
}
.step .dot {
    width: 28px; height: 28px; border-radius: 50%;
    display:flex; align-items:center; justify-content:center;
    background: rgba(120,165,230,.15); font-size: .8rem; font-weight: 700; color: var(--muted);
}
.step.done   { color: var(--ice); background: rgba(77,163,255,.09); border-color: rgba(77,163,255,.40); }
.step.done .dot { background: var(--blue); color: #04060C; }
.step.active {
    color: #fff; border-color: rgba(255,154,77,.55);
    background: linear-gradient(135deg, rgba(47,107,255,.35), rgba(255,106,61,.22));
    box-shadow: 0 0 26px -6px rgba(77,163,255,.7), 0 0 30px -14px rgba(255,154,77,.8);
}
.step.active .dot { background: linear-gradient(135deg, var(--blue), var(--ember)); color:#04060C; }

/* ---------- cards / containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border-color: var(--line) !important;
    background: linear-gradient(180deg, rgba(14,22,42,.78), rgba(8,13,26,.78)) !important;
    box-shadow: 0 18px 40px -26px rgba(0,0,0,.9);
}
.card-title { font-size: 1.15rem; font-weight: 700; margin: 0 0 2px 0; color: #fff; }
.card-sub   { color: var(--muted); font-size: .92rem; margin-bottom: 10px; }
.badge {
    display:inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: .72rem; font-weight: 700; letter-spacing: .06em;
    background: rgba(255,154,77,.14); color: var(--ember);
    border: 1px solid rgba(255,154,77,.35); margin-right: 8px;
}

/* ---------- inputs ---------- */
.stTextInput input, .stTextArea textarea {
    border-radius: 12px !important;
    background: #080E1C !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(77,163,255,.20) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: #55688A !important; }

/* ---------- buttons ---------- */
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #2F6BFF 0%, #4DA3FF 45%, #FF9A4D 130%) !important;
    border: none !important; color: #fff !important;
    border-radius: 12px !important; font-weight: 700 !important;
    padding: .55rem 1.2rem !important;
    box-shadow: 0 8px 24px -10px rgba(77,163,255,.9);
    transition: transform .15s ease, box-shadow .15s ease;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 30px -10px rgba(255,154,77,.85);
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] button {
    border-radius: 12px !important; font-weight: 600 !important;
    background: rgba(120,165,230,.07) !important;
    border: 1px solid var(--line) !important; color: var(--text) !important;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--ember) !important; color: #fff !important;
}
button:disabled { opacity: .4 !important; }

/* ---------- misc widgets ---------- */
[data-testid="stMetric"] {
    background: rgba(120,165,230,.06); border-radius: 14px; padding: 12px 16px;
    border: 1px solid var(--line);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--ember) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--ember) !important; }
[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, var(--blue), var(--ember)) !important; }
[data-testid="stExpander"] { border-color: var(--line) !important; border-radius: 14px !important; background: rgba(8,13,26,.6); }
[data-testid="stCode"] pre, pre { background: #060B16 !important; border: 1px solid var(--line); border-radius: 12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ───────────────────────────── Backend helpers ─────────────────────────────
@st.cache_resource
def get_graph():
    """Build the graph ONCE. The in-memory checkpointer must survive reruns."""
    return build_blog_graph()


def cfg() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def init_state():
    defaults = {
        "thread_id": None,
        "error": None,
        "research_round": 0,
        "topic_in": "",
        "audience_in": "Tech enthusiasts",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def read_stage():
    """Ask LangGraph where the run currently is. Returns (stage, payload, values)."""
    if not st.session_state.thread_id:
        return "input", {}, {}

    snap = get_graph().get_state(cfg())
    values = snap.values or {}
    if not values:  # server restarted / thread unknown
        st.session_state.thread_id = None
        return "input", {}, {}

    if snap.interrupts:
        payload = snap.interrupts[0].value
        if isinstance(payload, dict) and payload.get("state") in ("research_review", "draft_review"):
            return payload["state"], payload, values

    if not snap.next and values.get("final_blog"):
        return "done", {}, values

    return "stalled", {}, values


def run_graph(graph_input, spinner_text: str):
    """Invoke the graph (start or resume) with a spinner, then rerun the UI."""
    st.session_state.error = None
    try:
        with st.spinner(spinner_text):
            get_graph().invoke(graph_input, config=cfg())
    except Exception as e:  # noqa: BLE001 - show anything to the user
        st.session_state.error = f"{type(e).__name__}: {e}"
    st.rerun()


def start_run():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.research_round = 0
    run_graph(
        {
            "topic": st.session_state.topic_in.strip(),
            "audience": st.session_state.audience_in.strip() or "general readers",
        },
        "🔎 Researcher agent is building your outline…",
    )


def reset_run():
    st.session_state.thread_id = None
    st.session_state.error = None
    st.session_state.research_round = 0
    st.rerun()


# ───────────────────────────── UI pieces ─────────────────────────────
STEPS = ["Brief", "Review research", "Review draft", "Final blog"]
STAGE_INDEX = {"input": 0, "research_review": 1, "draft_review": 2, "stalled": 1, "done": 3}


def hero():
    st.markdown(
        '<div class="hero"><h1>✍️ Blog Studio</h1>'
        "<p>A team of AI agents researches, writes and edits your blog — "
        "and you stay in control at every checkpoint.</p></div>",
        unsafe_allow_html=True,
    )


def stepper(stage: str):
    current = STAGE_INDEX[stage]
    html = '<div class="stepper">'
    for i, name in enumerate(STEPS):
        if stage == "done" or i < current:
            cls, mark = "done", "✓"
        elif i == current:
            cls, mark = "active", str(i + 1)
        else:
            cls, mark = "", str(i + 1)
        html += f'<div class="step {cls}"><div class="dot">{mark}</div>{name}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def card_header(title: str, sub: str, badge: str | None = None):
    b = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="card-title">{b}{title}</div><div class="card-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def sidebar(stage: str, values: dict):
    with st.sidebar:
        st.markdown("## ✍️ Blog Studio")
        st.caption("Multi-agent blog writer · LangGraph + Groq")

        st.divider()
        st.markdown("**Pipeline**")
        st.markdown(
            "1. 🔎 **Researcher** drafts an outline\n"
            "2. 🙋 **You** approve or request changes\n"
            "3. ✍️ **Writer** produces the blog\n"
            "4. 🙋 **You** approve or request changes\n"
            "5. 🪄 **Editor** polishes the final post"
        )

        if values:
            st.divider()
            st.markdown("**Current session**")
            st.caption(f"Topic: {values.get('topic', '')}")
            st.caption(f"Audience: {values.get('audience', '')}")
            st.metric("Draft revisions", f"{values.get('revision_count', 0)} / {MAX_REVISIONS}")

        if st.session_state.thread_id:
            st.divider()
            if st.button("↺ Start over", use_container_width=True, key="reset_sidebar"):
                reset_run()


def view_input():
    with st.container(border=True):
        card_header("What should we write about?", "Give the agents a topic and tell them who will read it.", "STEP 1")

        st.text_input(
            "Blog topic",
            key="topic_in",
            placeholder="e.g. What is the Model Context Protocol (MCP)?",
        )
        st.text_input(
            "Target audience",
            key="audience_in",
            placeholder="e.g. Tech enthusiasts, college students, startup founders…",
        )

        st.caption("Need inspiration? Try one of these:")
        examples = ["What is GenAI?", "How do RAG systems work?", "Why LangGraph beats plain chains"]
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            col.button(
                ex,
                key=f"ex_{ex}",
                use_container_width=True,
                on_click=lambda e=ex: st.session_state.update(topic_in=e),
            )

        st.write("")
        disabled = not st.session_state.topic_in.strip()
        if st.button("🚀 Generate research outline", type="primary", disabled=disabled, key="start_btn"):
            start_run()


def render_content_tabs(text: str, key: str):
    preview, source = st.tabs(["👁️ Preview", "📝 Markdown"])
    with preview:
        st.markdown(text)
    with source:
        st.code(text, language="markdown")


def view_research(payload: dict, values: dict):
    rnd = st.session_state.research_round
    with st.container(border=True):
        card_header(
            "Review the research outline",
            "The Writer will build the blog from this outline. Approve it, or tell the Researcher what to change.",
            f"RESEARCH · ATTEMPT {rnd + 1}",
        )
        render_content_tabs(payload.get("research", ""), "research")

    with st.container(border=True):
        card_header("Your decision", "Leave feedback empty and approve, or describe what should change.")
        feedback = st.text_area(
            "Feedback for the Researcher",
            key=f"research_fb_{rnd}",
            placeholder="e.g. Add more real-world examples and focus on practical use cases.",
            height=110,
        )
        c1, c2, _ = st.columns([1, 1.2, 2])
        if c1.button("✅ Approve", type="primary", key="research_approve", use_container_width=True):
            run_graph(Command(resume={"action": "approve", "feedback": ""}), "✍️ Writer agent is drafting your blog…")
        if c2.button(
            "🔁 Request changes",
            key="research_revise",
            disabled=not feedback.strip(),
            use_container_width=True,
        ):
            st.session_state.research_round += 1
            run_graph(
                Command(resume={"action": "revise", "feedback": feedback.strip()}),
                "🔎 Researcher agent is revising the outline…",
            )


def view_draft(payload: dict, values: dict):
    used = values.get("revision_count", 0)
    limit_hit = used >= MAX_REVISIONS

    with st.container(border=True):
        card_header(
            "Review the draft",
            "This is what the Writer produced. Approve it to send it to the Editor, or request changes.",
            f"DRAFT · VERSION {used}",
        )
        st.progress(min(used / MAX_REVISIONS, 1.0), text=f"Writer attempts used: {used} / {MAX_REVISIONS}")
        render_content_tabs(payload.get("draft", ""), "draft")

        with st.expander("🔎 Research outline used for this draft"):
            st.markdown(values.get("research", ""))

    with st.container(border=True):
        card_header("Your decision", "Approve to send it to the Editor, or ask the Writer for a rewrite.")
        if limit_hit:
            st.warning(
                f"Revision limit ({MAX_REVISIONS}) reached — the Writer can't be asked again. "
                "Approve to send this draft to the Editor.",
                icon="🛑",
            )
        feedback = st.text_area(
            "Feedback for the Writer",
            key=f"draft_fb_{used}",
            placeholder="e.g. Make the intro punchier and add a real-world example in section 2.",
            height=110,
            disabled=limit_hit,
        )
        c1, c2, _ = st.columns([1, 1.2, 2])
        if c1.button("✅ Approve & edit", type="primary", key="draft_approve", use_container_width=True):
            run_graph(Command(resume={"action": "approve", "feedback": ""}), "🪄 Editor agent is polishing your blog…")
        if c2.button(
            "🔁 Request changes",
            key="draft_revise",
            disabled=limit_hit or not feedback.strip(),
            use_container_width=True,
        ):
            run_graph(
                Command(resume={"action": "revise", "feedback": feedback.strip()}),
                "✍️ Writer agent is rewriting the draft…",
            )


def view_done(values: dict):
    final_blog = values.get("final_blog", "")
    flag = f"celebrated_{st.session_state.thread_id}"
    if not st.session_state.get(flag):
        st.balloons()
        st.session_state[flag] = True

    m1, m2, m3 = st.columns(3)
    m1.metric("Words", f"{len(final_blog.split()):,}")
    m2.metric("Writer attempts", values.get("revision_count", 0))
    m3.metric("Audience", values.get("audience", "—"))

    with st.container(border=True):
        card_header("Your blog is ready", "Polished by the Editor agent.", "FINAL")
        render_content_tabs(final_blog, "final")

        st.write("")
        c1, c2, _ = st.columns([1.2, 1.2, 2])
        c1.download_button(
            "⬇️ Download .md",
            data=final_blog,
            file_name="blog_post.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
            key="download_btn",
        )
        if c2.button("✨ Write another", use_container_width=True, key="another_btn"):
            reset_run()

    with st.expander("📜 See earlier stages"):
        t1, t2 = st.tabs(["Research outline", "Writer draft (before editing)"])
        with t1:
            st.markdown(values.get("research", ""))
        with t2:
            st.markdown(values.get("draft", ""))


def view_stalled():
    st.warning(
        "The run stopped before finishing (probably an API error). "
        "Your progress is saved — you can retry from where it stopped.",
        icon="⏸️",
    )
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("🔄 Retry", type="primary", key="retry_btn", use_container_width=True):
        run_graph(None, "Resuming the run…")
    if c2.button("↺ Start over", key="reset_stalled", use_container_width=True):
        reset_run()


# ───────────────────────────── Main ─────────────────────────────
init_state()
stage, payload, values = read_stage()

sidebar(stage, values)
hero()
stepper(stage)

if st.session_state.error:
    st.error(st.session_state.error, icon="🚨")

if stage == "input":
    view_input()
elif stage == "research_review":
    view_research(payload, values)
elif stage == "draft_review":
    view_draft(payload, values)
elif stage == "done":
    view_done(values)
else:
    view_stalled()