"""Streamlit Web UI for JARVIS - Chat interface with agent system."""
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure page
st.set_page_config(
    page_title="JARVIS - AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .sidebar-section {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "jarvis_initialized" not in st.session_state:
    st.session_state.jarvis_initialized = False
if "agent_core" not in st.session_state:
    st.session_state.agent_core = None
if "monitor_agent" not in st.session_state:
    st.session_state.monitor_agent = None
if "designer_agent" not in st.session_state:
    st.session_state.designer_agent = None
if "spawner" not in st.session_state:
    st.session_state.spawner = None
if "use_agent" not in st.session_state:
    st.session_state.use_agent = True


def run_async(coro):
    """Run async coroutine in event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def initialize_jarvis():
    """Initialize JARVIS components."""
    from jarvis.agent import (
        get_agent_core, get_monitor_agent, get_designer_agent,
        get_spawner
    )

    # Initialize core components
    agent_core = get_agent_core()
    await agent_core.initialize()

    monitor = get_monitor_agent()
    await monitor.initialize()

    designer = get_designer_agent()
    await designer.initialize()

    spawner = get_spawner()
    await spawner.initialize()

    return {
        "agent_core": agent_core,
        "monitor": monitor,
        "designer": designer,
        "spawner": spawner,
    }


def initialize_session():
    """Initialize JARVIS in session state."""
    if not st.session_state.jarvis_initialized:
        with st.spinner("Initialisation de JARVIS..."):
            try:
                components = run_async(initialize_jarvis())
                st.session_state.agent_core = components["agent_core"]
                st.session_state.monitor_agent = components["monitor"]
                st.session_state.designer_agent = components["designer"]
                st.session_state.spawner = components["spawner"]
                st.session_state.jarvis_initialized = True
                st.session_state.llm_available = run_async(check_llm_available())
                st.success("JARVIS initialise avec succes !")
            except Exception as e:
                st.error(f"Erreur d'initialisation: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.jarvis_initialized = False


async def check_llm_available() -> bool:
    """Check if any LLM provider is reachable."""
    try:
        from jarvis.llm import get_llm_engine
        llm = get_llm_engine()
        response = await llm.complete("Say OK", system_prompt="Reply with just OK")
        return bool(response)
    except Exception:
        return False


async def process_user_message(text: str, use_agent: bool = True) -> str:
    """Process user message through JARVIS with smart routing."""
    agent_core = st.session_state.agent_core
    llm_available = st.session_state.get("llm_available", False)

    # Step 1: Check if it matches a known intent (system commands, files, etc.)
    from jarvis.intent import get_intent_recognizer, get_action_executor
    recognizer = get_intent_recognizer()
    executor = get_action_executor()
    intent = await recognizer.recognize(text)

    # If intent has high confidence and is NOT chat, execute it directly
    if intent.confidence >= 0.7 and intent.name != "chat":
        return await executor.execute_intent(intent)

    # Step 2: If LLM available, use direct chat (faster than agent for small models)
    if llm_available:
        from jarvis.llm import get_llm_engine
        llm = get_llm_engine()
        try:
            response = await llm.complete(
                text,
                system_prompt="Tu es JARVIS, un assistant IA personnel inspire d'Iron Man. Reponds en francais, de maniere concise et utile. Si la tache necessite des outils (fichiers, shell, etc.), reponds DIRECT avec ce que tu ferais."
            )
            return response
        except Exception as e:
            return f"Erreur LLM: {e}"

    # Step 3: Fallback to intent executor (handles chat, system commands without LLM)
    return await executor.process_text(text)


def send_message(text: str, use_agent: bool = True):
    """Send user message and get response."""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "timestamp": datetime.now()
    })

    # Get response
    with st.spinner("JARVIS reflechit..."):
        try:
            response = run_async(process_user_message(text, use_agent))
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
        except Exception as e:
            import traceback
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Erreur: {str(e)}\n\n```\n{traceback.format_exc()}\n```",
                "timestamp": datetime.now()
            })


def render_sidebar():
    """Render sidebar with controls and status."""
    with st.sidebar:
        st.markdown('<div class="main-header"><h2>JARVIS Control</h2></div>', unsafe_allow_html=True)

        # Status
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Statut")

        if st.session_state.jarvis_initialized:
            st.success("JARVIS: Actif")

            # LLM status
            llm_ok = st.session_state.get("llm_available", False)
            if llm_ok:
                st.success("LLM: Disponible (Agent ReAct)")
            else:
                st.warning("LLM: Indisponible (Mode Intentions)")

            # Agent core status
            if st.session_state.agent_core:
                try:
                    status = st.session_state.agent_core.get_status()
                    st.info(f"Etat: {status['state']}")
                    if status.get('task'):
                        task = status['task']
                        st.write(f"Tache: {task.get('goal', '')[:50]}...")
                        st.write(f"Iterations: {task.get('iterations', 0)}")
                except Exception as e:
                    st.warning(f"Erreur statut: {e}")
        else:
            st.warning("JARVIS: Non initialise")
        st.markdown('</div>', unsafe_allow_html=True)

        # Controls
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Controles")

        st.session_state.use_agent = st.checkbox(
            "Mode Agent (ReAct)",
            value=st.session_state.use_agent,
            help="Utiliser le systeme agentique complet"
        )

        if st.button("Reinitialiser JARVIS"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if st.button("Effacer l'historique"):
            st.session_state.messages = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick Actions
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Actions Rapides")

        quick_actions = [
            ("Code", "Cree une fonction Python factorielle"),
            ("Arch", "Concois une architecture microservices"),
            ("Monitor", "Montre l'etat systeme"),
            ("Reflect", "Analyse cette architecture"),
            ("Fichier", "Liste mes documents"),
        ]

        for label, prompt in quick_actions:
            if st.button(label, use_container_width=True, key=f"qa_{label}"):
                send_message(prompt)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Monitor Metrics
        if st.session_state.monitor_agent:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.subheader("Metriques Systeme")
            if st.button("Actualiser"):
                try:
                    metrics = run_async(st.session_state.monitor_agent.get_status())
                    if metrics.get('current_metrics'):
                        m = metrics['current_metrics']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("CPU", f"{m.get('cpu_percent', 0):.1f}%")
                            st.metric("RAM", f"{m.get('memory_percent', 0):.1f}%")
                        with col2:
                            st.metric("Disque", f"{m.get('disk_percent', 0):.1f}%")
                            st.metric("Processus", m.get('process_count', 0))
                except Exception as e:
                    st.error(f"Erreur: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Sub-agents
        if st.session_state.spawner:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.subheader("Sous-Agents")
            try:
                agents = st.session_state.spawner.list_agents()
                if agents:
                    for agent in agents:
                        st.write(f"- {agent['id']} ({agent['type']})")
                        st.caption(f"  {agent['goal'][:40]}... [{agent['state']}]")
                else:
                    st.caption("Aucun sous-agent actif")
            except Exception as e:
                st.caption(f"Erreur: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


def render_chat():
    """Render chat interface."""
    st.markdown('<div class="main-header"><h1>Chat JARVIS</h1><p>Parlez-moi naturellement - je comprends le francais</p></div>', unsafe_allow_html=True)

    # Chat container
    chat_container = st.container()

    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input area
    st.markdown("---")

    user_input = st.text_input(
        "Votre message:",
        placeholder="Tapez votre message... (ex: 'Cree un fichier test.py')",
        key="user_input",
    )

    if user_input and st.session_state.get("last_input") != user_input:
        st.session_state.last_input = user_input
        send_message(user_input, use_agent=st.session_state.use_agent)
        st.rerun()


def render_agent_tools():
    """Render agent-specific tools tabs."""
    tab1, tab2, tab3, tab4 = st.tabs(["Designer", "Monitor", "Reflector", "Spawner"])

    with tab1:
        st.subheader("Generation de Code / Architecture")
        col1, col2 = st.columns([3, 1])
        with col1:
            spec = st.text_area("Specification:",
                               placeholder="Ex: Cree une API REST FastAPI pour gerer des taches avec auth JWT",
                               height=100)
        with col2:
            dtype = st.selectbox("Type", ["code", "architecture", "docs", "review", "refactor", "scaffold"])
            lang = st.selectbox("Langage", ["python", "javascript", "typescript", "rust", "go"])

        if st.button("Generer", type="primary", disabled=not spec):
            with st.spinner("Generation en cours..."):
                from jarvis.agent import DesignTask
                task = DesignTask(type=dtype, spec=spec, language=lang)
                try:
                    if dtype == "code":
                        result = run_async(st.session_state.designer_agent.generate_code(task))
                    else:
                        result = run_async(st.session_state.designer_agent.design_architecture(task))

                    if result.get("code"):
                        st.code(result["code"], language=lang)
                    elif result.get("architecture"):
                        st.markdown(result["architecture"])
                    if result.get("saved_to"):
                        st.success(f"Sauvegarde: {result['saved_to']}")
                except Exception as e:
                    st.error(f"Erreur: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("Surveillance Systeme")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Health Check Complet"):
                with st.spinner("Verification..."):
                    try:
                        health = run_async(st.session_state.monitor_agent.run_health_check())
                        st.json(health)
                    except Exception as e:
                        st.error(f"Erreur: {e}")
        with col2:
            if st.button("Metriques Temps Reel"):
                try:
                    metrics = run_async(st.session_state.monitor_agent.get_status())
                    st.json(metrics)
                except Exception as e:
                    st.error(f"Erreur: {e}")

    with tab3:
        st.subheader("Reflexion Profonde")
        topic = st.text_input("Sujet de reflexion:")
        depth = st.slider("Profondeur", 1, 5, 3)
        context = st.text_area("Contexte (optionnel):")

        if st.button("Reflechir", disabled=not topic):
            with st.spinner("Reflexion en cours..."):
                try:
                    from jarvis.agent import get_reflector_agent
                    reflector = get_reflector_agent()
                    reflection = run_async(reflector.reflect(topic, context or None, depth))
                    st.markdown("### Critique")
                    st.write(reflection.critique)
                    st.markdown("### Ameliorations")
                    for i, imp in enumerate(reflection.improvements, 1):
                        st.write(f"{i}. {imp}")
                except Exception as e:
                    st.error(f"Erreur: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab4:
        st.subheader("Gestion des Sous-Agents")
        col1, col2 = st.columns(2)
        with col1:
            goal = st.text_area("Objectif du sous-agent:")
            atype = st.selectbox("Type", ["generic", "designer", "monitor", "reflector"])
        with col2:
            timeout = st.number_input("Timeout (s)", value=300, min_value=30)

        if st.button("Lancer Sous-Agent", disabled=not goal):
            from jarvis.agent import SubAgentSpec, AgentType
            spec = SubAgentSpec(
                name=f"web_agent_{datetime.now().strftime('%H%M%S')}",
                agent_type=AgentType(atype),
                goal=goal,
                timeout=timeout
            )
            with st.spinner("Execution..."):
                try:
                    agent = run_async(st.session_state.spawner.spawn_and_wait(spec))
                    if agent.result:
                        st.success(f"Termine: {agent.result[:500]}")
                    if agent.error:
                        st.error(f"Erreur: {agent.error}")
                except Exception as e:
                    st.error(f"Erreur: {e}")
                    import traceback
                    st.code(traceback.format_exc())


def main():
    """Main Streamlit app."""
    # Initialize
    initialize_session()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>JARVIS - Assistant IA Personnel</h1>
        <p>Inspire d'Iron Man | Architecture Agentique | Local-First</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        render_chat()

    with col2:
        if st.session_state.jarvis_initialized:
            render_agent_tools()
        else:
            st.info("Initialisez JARVIS depuis la barre laterale")


if __name__ == "__main__":
    main()
