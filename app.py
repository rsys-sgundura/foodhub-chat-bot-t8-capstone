
import streamlit as st
import base64
import sys
from validate_customer import is_valid_customer
from agent.chat_agent import answer_tool_func
from agent.sql_agent import order_query_tool_func
from agent.llm_models import llm_model
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# ================================================================
#  12.7 : CHAT AGENT INITIALIZATION
# ---------------------------------------------------------------
#  PURPOSE:
#   - Combine OrderQueryTool + AnswerTool into a single LangChain agent.
#   - Enable seamless, conversational query handling for FoodHub.
#   - Provide database lookups + natural language formatting.
#   - Uses a deterministic LLM for concise, factual responses.
# ================================================================


# ----------------------------------------------------------------
#  Step 1: Create LangChain Tool for SQL Order Query Function
# ----------------------------------------------------------------
#  This tool wraps the low-level `order_query_tool_func` which
#  interacts with the database to retrieve order info.
#  The tool’s description guides the LLM on when to invoke it.
# =================================================================
OrderQueryTool = Tool(
    name="order_query_tool",
    func=order_query_tool_func,
    description=(
        "Use this tool to fetch order-related (read-only) information "
        "for a customer's order using their customer ID. "
        "It filters confidential fields and returns a structured stringified dictionary."
    )
)


# ----------------------------------------------------------------
#  Step 2: Create LangChain Tool for Answer Formatting Function
# ----------------------------------------------------------------
#  This tool wraps the `answer_tool_func` which converts raw
#  database responses into polite, user-friendly chatbot replies.
#  It enforces escalation, cancellation, and order-status policies.
# ================================================================
AnswerTool = Tool(
    name="answer_tool",
    func=answer_tool_func,
    description=(
        "Format raw DB results into concise, polite, and user-facing messages. "
        "Handles cancellation, delivery status, and escalation logic."
    )
)

# ----------------------------------------------------------------
#  Step 3: Register Active LangChain Tools
# ----------------------------------------------------------------
#  Defines the toolset that the agent can use for reasoning.
#   - OrderQueryTool : Handles SQL data fetches
#   - AnswerTool     : Generates human-like order updates
# ================================================================
tools = [OrderQueryTool, AnswerTool]

# ----------------------------------------------------------------
#  Step 4: Initialize Chat Agent
# ----------------------------------------------------------------
#  Configures a ZERO_SHOT_REACT_DESCRIPTION type agent:
#   - Zero-shot: Can reason without prior examples.
#   - React-style: Uses internal reasoning traces.
#   - Description-driven: Selects tools via their descriptions.
#  This agent can autonomously decide which tool to call next.
# ================================================================
chat_agent = initialize_agent(
    tools=tools,                                # Registered functional tools
    llm=llm_model,                              # Underlying Groq-based LLM
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent reasoning mode
    verbose=False                               # Suppress detailed logs
)

# (Optional) You can limit tool chaining depth if needed
# chat_agent.max_iterations = 1

# ================================================================
#  AGENT CONTROLLER
# ---------------------------------------------------------------
#  PURPOSE:
#   - Manages the logical execution flow between tools.
#   - Ensures correct sequential invocation:
#       1️⃣ OrderQueryTool → Fetch order info from DB.
#       2️⃣ AnswerTool     → Convert data to user-friendly response.
# ================================================================

# ------------------------------------------------
#  Step 5: Define agent_response()
# ------------------------------------------------
#  PURPOSE:
#   - Acts as orchestrator for tool execution.
#   - Builds a structured internal instruction prompt.
#   - Ensures deterministic pipeline flow:
#       OrderQueryTool → AnswerTool → Return response.
# ------------------------------------------------
def agent_response(cust_id: str, user_query: str) -> str:
    """
    Executes the end-to-end toolchain:
    1️⃣ Query DB using OrderQueryTool.
    2️⃣ Generate response using AnswerTool.
    Returns the final formatted chatbot message.
    """

    # === Construct Internal Execution Prompt ===
    agent_prompt = f"""
    You are FoodHub’s Order Assistant.
    Follow this strict tool usage sequence:
    1️⃣ Call 'OrderQueryTool' with:
        input_string = str({{"cust_id": "{cust_id}", "user_query": "{user_query}"}})
       → Output: A stringified dictionary containing 'cust_id', 'orig_query', and 'db_orders'.
    2️⃣ Then call 'AnswerTool' with:
        input_string = the exact output received from OrderQueryTool.
    3️⃣ Finally, return **only** the AnswerTool’s response — do not paraphrase or summarize.
    """
    # === Execute the Prompt via Agent ===
    final_answer = chat_agent.run(agent_prompt)

    # === Return Final Answer ===
    return final_answer

# ------------------------------------------------
#  Step 6: Define chatbot_response()
# ------------------------------------------------
#  PURPOSE:
#   - Entry point for Streamlit UI.
#   - Accepts `cust_id` and natural `user_query`.
#   - Passes data to the LangChain agent pipeline.
#   - Returns LLM-generated final message for display.
# ------------------------------------------------
def chatbot_response(cust_id: str, user_query: str) -> str:
    final_llm_response = agent_response(cust_id, user_query)
    return final_llm_response

# ================================================================
#  STREAMLIT USER INTERFACE CONFIGURATION
# ---------------------------------------------------------------
#  PURPOSE:
#   - Build the frontend chat interface for FoodHub.
#   - Handle authentication, background visuals, and chat layout.
# ================================================================

# --- Step 1: Basic App Configuration ---
st.set_page_config(page_title="FoodHub Chatbot", page_icon="🍽️", layout="wide")


# --- Step 2: Utility: Load Background Image ---
def get_base64_image(image_path):
    """Reads local image and returns Base64-encoded string for CSS embedding."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# --- Step 3: Initialize Session State Variables ---
image_base64 = get_base64_image("foodhub_background_jpg.jpg")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================================================================
#  AUTHENTICATION LOGIC
# ---------------------------------------------------------------
#  - Displays login form if user not authenticated.
#  - Validates credentials using `is_valid_customer()`.
#  - Once validated, reloads app with chat interface.
# ================================================================
if not st.session_state.authenticated:
    col1, col2 = st.columns([2, 2])
    with col2:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("foodhub_logo.png", width=500)
        with col2:
            st.markdown("<h1 style='color: #ff4b4b;'>Welcome to FoodHub Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: black;'>Please enter customer ID and password to continue</p>", unsafe_allow_html=True)

        # --- Login Form ---
        with st.form("login_form"):
            customer_id = st.text_input("Customer ID", placeholder="eg: C1018")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if is_valid_customer(customer_id) and password == "foodhub123":
                    st.session_state.authenticated = True
                    st.session_state.customer_id = customer_id
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

# ================================================================
#  CHATBOT MAIN INTERFACE
# ---------------------------------------------------------------
#  - Loads only after successful authentication.
#  - Displays chat history, bubbles, and input box.
#  - Calls `chatbot_response()` for each user query.
# ================================================================
if st.session_state.authenticated:
    customer_id = st.session_state.get("customer_id")

    # --- Initialize Chat History (Welcome Message) ---
    if not st.session_state.chat_history:
        st.session_state["chat_history"] = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

    spacer_left, chat_col, spacer_right = st.columns([2, 4, 1])

    # --- Header and Logout Section ---
    with chat_col:
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.image("foodhub_logo.png", width=100)
        with col2:
            st.markdown(f"<h1 style='color: #ff4b4b;'>Hey {customer_id}, Welcome!</h1>", unsafe_allow_html=True)
        with col3:
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.customer_id = None
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("---")

        # --- Chat Bubble Styles ---
        st.markdown("""
        <style>
        .chat-bubble-user {
            background-color: #dfe6e9;
            color: #000000;
            padding: 10px 14px;
            border-radius: 12px;
            margin-bottom: 6px;
            max-width: 80%;
            text-align: right;
        }
        .chat-bubble-bot {
            background-color: #f1f0f0;
            color: #000000;
            padding: 10px 14px;
            border-radius: 12px;
            margin-bottom: 6px;
            max-width: 80%;
            text-align: left;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- Render Chat History ---
        for m in st.session_state.chat_history:
            left_col, right_col = st.columns([1, 1])
            if m["role"] == "user":
                with right_col:
                    st.markdown(f"""
                        <div style='display:flex; justify-content:flex-end; align-items:center; margin-bottom:8px;'>
                            <div class='chat-bubble-user'>{m['content']}</div>
                            <div style='font-size:20px; margin-left:8px;'>🙋</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                with left_col:
                    st.markdown(f"""
                        <div style='display:flex; justify-content:flex-start; align-items:center; margin-bottom:8px;'>
                            <div style='font-size:20px; margin-right:8px;'>🤖</div>
                            <div class='chat-bubble-bot'>{m['content']}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # --- User Input Section ---
        user_input = st.chat_input("Ask about your order or menu...")

        if user_input:
            # Step 1️⃣: Append user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Step 2️⃣: Fetch response from chatbot pipeline
            with st.spinner("Let me check that for you..."):
                final_response = chatbot_response(customer_id, user_input)
                print("Output of chatbot_response tool:", final_response, flush=True)
                sys.stdout.flush()

            # Step 3️⃣: Append assistant response and rerun
            st.session_state.chat_history.append({"role": "assistant", "content": final_response})
            st.rerun()
