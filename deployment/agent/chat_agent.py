# 12.2
# ================================================================
#  FILE: chat_agent.py
# ---------------------------------------------------------------
#  FoodHub Conversational Assistant (Groq-exclusive version)
# ---------------------------------------------------------------
#  PURPOSE:
#   - Handles all user-facing chat interactions for FoodHub.
#   - Uses Groq-hosted LLaMA 4 model for short (<80 words), polite,
#     and context-aware responses.
#   - Detects intent (promo, refund, handoff, farewell, etc.)
#     and responds accordingly.
#   - Enforces data privacy and safety policies.
# ================================================================

import os
import re
import streamlit as st
import sys
from langchain_groq import ChatGroq

from langchain.agents import initialize_agent, Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.agent_types import AgentType

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ================================================================
#  SECTION 1: LLM Initialization (Low Temperature)
# ---------------------------------------------------------------
#  Purpose:
#    Sets up a deterministic Groq-powered Large Language Model (LLM)
#    with low temperature (0.0) for predictable and consistent outputs.
#    Fetches the API key securely from Streamlit secrets or environment
#    variables and stops execution if missing.
# ================================================================

@st.cache_resource
def initialize_llm_high():
    """
    Initialize the Groq-based LLM with high creativity (temperature = 0.7).
    
    Workflow:
    1️⃣ Retrieve Groq API key (from Streamlit secrets or environment variable).
    2️⃣ Validate key existence; stop execution if not found.
    3️⃣ Configure and return a ChatGroq instance for deterministic responses.
    """

    # ------------------------------------------------------------
    # Step 1: Retrieve Groq API Key
    # Attempt to load the API key securely from Streamlit secrets;
    # if not found, fallback to system environment variable.
    # ------------------------------------------------------------
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except:
        groq_api_key = os.getenv("GROQ_API_KEY")

    # ------------------------------------------------------------
    # Step 2: Validate API Key
    # If the key is missing, display a helpful error message
    # and stop further execution to prevent runtime failures.
    # ------------------------------------------------------------
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY Environment Variable Not Found! Please set the environment variable.")
        st.info("Please create a `.streamlit/secrets.toml` file with:\n```\nGROQ_API_KEY = \"your-api-key-here\"\n```")
        st.stop()

    # ------------------------------------------------------------
    # Step 3: Configure and Initialize Groq LLM
    # Create a ChatGroq instance using a high-temperature setup
    # for Conversational and natural sounding responses.
    # ------------------------------------------------------------
    llmh = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq-hosted LLaMA model
        temperature=0.7,                                      # High temperature → Conversational output
        max_tokens=200,                                     # Limit response size
        max_retries=0,                                      # No automatic retries
        groq_api_key=groq_api_key                           # Secure API key injection
    )

    # ------------------------------------------------------------
    # Step 4: Return Cached LLM Instance
    # The LLM object is cached to avoid reinitialization overhead.
    # ------------------------------------------------------------
    return llmh


# ================================================================
#  SECTION 2: Create Global LLM Instance
# ---------------------------------------------------------------
#  Initializes the cached High-temperature LLM for consistent use
#  across the Streamlit app pipeline for conversational response.
# ================================================================
llm_high = initialize_llm_high()


# ================================================================
#  SECTION 3: Escalation Detection
# ---------------------------------------------------------------
#  Purpose:
#    Identifies user queries that indicate unresolved issues, 
#    urgency, dissatisfaction, or explicit requests to speak 
#    with a human support representative.  
#    Helps route critical or frustrated customer messages 
#    to human agents for faster resolution.
# ================================================================

def check_escalation(user_query: str) -> str:
    """
    Detects whether a user's message requires escalation to human support.
    Logic:
    - Scans the user query for specific keywords or phrases that suggest:
        * Repeated complaints or unresolved issues.
        * Requests for urgent or immediate attention.
        * Direct mentions of escalation, dissatisfaction, or need for human help.
    - Returns:
        * "Escalated"      → if any escalation keyword is detected.
        * "Not Escalated"  → if no escalation indicators are present.
    """

    # ------------------------------------------------------------
    # Step 1: Define escalation-related keywords and phrases
    # These capture user frustration, urgency, or explicit escalation intent.
    # ------------------------------------------------------------
    escalation_kw_list = [
        "issue persists", "not resolved", "complaint", "contact human",
        "priority", "immediate", "service failure", "speak to manager",
        "support required", "help me now", "not satisfied", "request escalation",
        "critical issue", "issue unresolved", "need assistance", "escalation",
        "problem still exists", "no response", "cannot resolve", "urgent",
        "multiple times", "immediate response", "problem", "escalate",
        "still not working"
    ]

    # ------------------------------------------------------------
    # Step 2: Check for escalation triggers in the user’s query
    # Perform a case-insensitive match of any keyword in the query text.
    # ------------------------------------------------------------
    if any(keyword in user_query.lower() for keyword in escalation_kw_list):
        return "Escalated"   # 🚨 Escalation required — route to human support

    # ------------------------------------------------------------
    # Step 3: No escalation keywords found — proceed normally
    # ------------------------------------------------------------
    return "Not Escalated"


# ================================================================
#  SECTION 4: Order Cancellation Handler
# ---------------------------------------------------------------
#  Purpose:
#    Processes and validates customer cancellation requests 
#    based on the current order status.  
#    Ensures cancellations are not permitted for orders that 
#    are already delivered, canceled, or beyond the preparation stage.
# ================================================================

def handle_cancellation(user_query: str, raw_orders: str, order_status: str) -> str:
    """
    Handles customer order cancellation requests logically and politely.
    Logic:
    - Identifies if the user’s message contains a cancellation intent.
    - Evaluates the current order status and determines whether cancellation 
      is still possible.
    - Returns a context-appropriate message explaining the outcome.
    """

    # ------------------------------------------------------------
    # Step 1: Detect cancellation intent in the user’s query
    # If the message doesn’t contain the word “cancel”, skip processing.
    # ------------------------------------------------------------
    if "cancel" not in user_query.lower():
        return ""

    # ------------------------------------------------------------
    # Step 2: Check if order is already completed or canceled
    # In such cases, cancellation cannot be performed again.
    # ------------------------------------------------------------
    if order_status and order_status.lower() in ["delivered", "canceled"]:
        return (
            f"Your order has already been {order_status.lower()}. "
            "Cancellation is therefore not possible. We appreciate your understanding!"
        )

    # ------------------------------------------------------------
    # Step 3: Check if order is already being prepared or picked up
    # Once food preparation or pickup starts, cancellations are disallowed.
    # ------------------------------------------------------------
    elif order_status and order_status.lower() in ["preparing food", "picked up"]:
        return (
            f"Your order is currently {order_status.lower()}. "
            "Unfortunately, cancellations are not permitted at this stage. Thank you for your understanding!"
        )

    # ------------------------------------------------------------
    # Step 4: Default case — cancellation not allowed for unspecified reasons
    # ------------------------------------------------------------
    else:
        return (
            "Your order cannot be canceled at this moment. "
            "We appreciate your patience and look forward to serving you again!"
        )

# ================================================================
#  SECTION 5: Answer Tool — Final Response Generator
# ---------------------------------------------------------------
#  Purpose:
#    Processes the structured output from `OrderQueryTool`,
#    interprets order details, applies escalation or cancellation logic,
#    and generates a natural, customer-friendly response using the LLM.
# ================================================================

# ----------------------------------------------------------------
#  Function: answer_tool_func()
#  Description:
#    - Receives a stringified dictionary from the previous tool.
#    - Parses and validates it.
#    - Checks for escalation or cancellation triggers.
#    - Uses the LLM to craft the final user-facing message.
# ----------------------------------------------------------------
def answer_tool_func(answertool_input: str) -> str:
    """
    Receives the output from OrderQueryTool as stringified dict,
    parses it, and generates the final friendly message.
    """
    # ------------------------------------------------------------
    # Step 1: Parse the input dictionary safely
    # ------------------------------------------------------------
    try:
        data = ast.literal_eval(answertool_input)
        cust_id = data.get("cust_id", "Unknown")
        user_query = data.get("orig_query", "")
        db_response = data.get("db_response", "No order details found.")
    except Exception:
        # Handle invalid or malformed data gracefully
        return "⚠️ Error: Could not parse order data properly."

    # Initialize key order-related variables
    order_status = None
    item_in_order = None
    preparing_eta = None
    delivery_time = None
    
    print('answer_tool_func : LEVEL-1 Done',flush=True)
    print('cust_id = ',cust_id, flush=True)
    print('orig_query = ',user_query, flush=True)
    print('db_response = ',db_response, flush=True)
    sys.stdout.flush()
    
    # ------------------------------------------------------------
    # Step 2: Extract order details from db_response text
    # ------------------------------------------------------------
    for line in db_response.splitlines():
        if "Order Status" in line:
            order_status = line.split(":", 1)[1].strip()
        elif "Preparing ETA" in line:
            preparing_eta = line.split(":", 1)[1].strip()
        elif "Delivery Time" in line:
            delivery_time = line.split(":", 1)[1].strip()

    # ------------------------------------------------------------
    # Step 3: Detect if query needs escalation (critical or unresolved issues)
    # ------------------------------------------------------------
    escalation_var = check_escalation(user_query)
    if escalation_var == "Escalated":
        return (
            f"The current status of your order is: {order_status.lower()}. " +
            "⚠️ This issue needs urgent attention. " +
            "Your request has been escalated to a human support agent who will reach out to you soon."
        )

    #print('answer_tool_func : LEVEL-2 Done',flush=True)
    #sys.stdout.flush()
    
    # ------------------------------------------------------------
    # Step 4: Check for order cancellation requests
    # ------------------------------------------------------------
    cancel_response = handle_cancellation(user_query, db_response, order_status)
    if cancel_response:  # Return cancellation message if applicable
        return cancel_response

    #print('answer_tool_func : LEVEL-3 Done',flush=True)
    #sys.stdout.flush()

    #return "Forced: Thank you and your order conatins Steak..!"
    
    # ------------------------------------------------------------
    # Step 5: Build the system prompt for LLM to interpret and respond
    # ------------------------------------------------------------
    system_prompt = f"""
    You are a warm and helpful customer support assistant for FoodHub.
    Customer ID: {cust_id}
    Below is the customer's order information retrieved from the database:
    {db_response}
    Sample raw_orders format:
    order_id: O12493,
    cust_id: C1018,
    order_time: 12:35,
    order_status: picked up,
    payment_status: COD,
    item_in_order: Steak,
    preparing_eta: 12:50,
    prepared_time: 12:50,
    delivery_eta: 1:10,
    delivery_time: None
    Response Instructions:
    1. Respond in a friendly, natural, and concise tone — keep replies short.
    2. Use only the details from `db_response`. Do not infer or create extra info.
    3. Convert database text into polite, human-readable responses.
    4. When order_status = 'preparing food':
       - Include both 'preparing_eta' and 'delivery_eta'.
       - If 'delivery_eta' is missing or None, say: "Your order is being prepared, and the delivery ETA will be available soon."
    5. When order_status = 'delivered', include 'delivery_time' in the message.
    6. When order_status = 'canceled', explain politely and empathetically.
    7. When order_status = 'picked up':
       - Include 'delivery_eta' if available.
       - If 'delivery_eta' is missing or None, say: "Your order has been picked up, and the delivery ETA will be available soon."
    8. If the user query contains “Where is my order”, include the current 'order_status'.
    9. If the user query includes “How many items”, count the 'item_in_order' list and reply like:
       "Your order includes 3 items."
    """

    # ------------------------------------------------------------
    # Step 6: Build and send user-specific prompt to LLM
    # ------------------------------------------------------------
    user_prompt = f"User Query: {user_query}"

    # Generate final response using the configured LLM
    response_msg = llm_high.predict_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # ------------------------------------------------------------
    # Step 7: Clean and finalize the LLM response
    # ------------------------------------------------------------
    response = response_msg.content.strip()

    #print('answer_tool_func : LEVEL-4 Done; response = ',response, flush=True)
    #sys.stdout.flush()
    
    # Provide fallback message in case of empty or invalid response
    if not response:
        return "Sorry, we could not extract your order details at this time. Please try again later.."

    # Return the final generated response
    return response
       

# ================================================================
#  SECTION 6: LangChain Tool Wrapper
# ---------------------------------------------------------------
#  Wraps the chat handler as a LangChain Tool so that it can be
#  called within multi-agent workflows or pipelines.
# ================================================================
#AnswerTool = Tool(
#    name="answer_tool",
#    func=answer_tool_func,
#    description="Format raw DB results into a brief, polite user-facing message. Enforces business rules (cancelled/completed messaging, escalation)."
#)
