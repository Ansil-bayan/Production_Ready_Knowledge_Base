import os
import requests
from dotenv import load_dotenv

load_dotenv()


VAPI_API_KEY = os.getenv("VAPI_API_KEY")
SERVER_URL = os.getenv("SERVER_URL")
BUSINESS_ACTION_URL = os.getenv("BUSINESS_ACTION_URL") # Webhook for Optional Business Action
HUMAN_AGENT_PHONE = os.getenv("HUMAN_AGENT_PHONE") # Number for human escalation

headers = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json"
}

# 1. Knowledge Base Search Tool
kb_tool_payload = {
    "type": "function",
    "async": False,
    "function": {
        "name": "search_knowledge_base",
        "description": "Searches the business_loans_knowledge_base.json for FAQs, interest rates, policies, and loan terms.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query derived from user request."}
            },
            "required": ["query"]
        }
    },
    "server": {"url": SERVER_URL}
}

# 2. Optional Business Action Tool: Record Lead & Eligibility Check
lead_tool_payload = {
    "type": "function",
    "async": False,
    "function": {
        "name": "submit_lead_qualification",
        "description": "Submits qualified lead details to the CRM and schedules a call back.",
        "parameters": {
            "type": "object",
            "properties": {
                "business_name": {"type": "string"},
                "monthly_revenue": {"type": "number"},
                "years_in_business": {"type": "number"},
                "credit_score": {"type": "number"},
                "loan_amount_requested": {"type": "number"}
            },
            "required": ["business_name", "monthly_revenue", "years_in_business", "credit_score"]
        }
    },
    "server": {"url": BUSINESS_ACTION_URL}
}

# 3. Native Vapi Transfer Call Tool
transfer_tool_payload = {
    "type": "transferCall",
    "destinations": [
        {
            "type": "number",
            "number": HUMAN_AGENT_PHONE,
            "description": "Transfer to a human loan specialist"
        }
    ],
    "messages": [
        {
            "type": "request-start",
            "content": "I am transferring you to a human specialist now."
        }
    ]
}

# Register Tools in Vapi
kb_tool_res = requests.post("https://api.vapi.ai/tool", json=kb_tool_payload, headers=headers).json()
lead_tool_res = requests.post("https://api.vapi.ai/tool", json=lead_tool_payload, headers=headers).json()
transfer_tool_res = requests.post("https://api.vapi.ai/tool", json=transfer_tool_payload, headers=headers).json()

kb_tool_id = kb_tool_res["id"]
lead_tool_id = lead_tool_res["id"]
transfer_tool_id = transfer_tool_res["id"]

# 4. Define Business Loan Qualification System Prompt & Rules
SYSTEM_PROMPT = """
You are an AI Voice Agent representing Apex Capital for Business Loans.

--- CONVERSATION FLOW ---
1. GREETING & INTENT: Welcome the caller and ask how much financing they are seeking.
2. QUALIFICATION STEPS: Collect the following 4 data points sequentially:
   - Annual/Monthly Revenue
   - Time in Business (Years/Months)
   - Credit Score estimate
   - Desired Loan Amount

3. KNOWLEDGE BASE GROUNDING & FALLBACK RULES:
   - When answering customer questions about rates, policies, eligibility, or objections, execute `search_knowledge_base`.
   - QUERY GENERATION RULE: Make the search query specific and descriptive using exact nouns from the caller's request.
   - QUERY FORMULATION RULE: When calling `search_knowledge_base`, expand the query parameter into a full descriptive sentence (e.g., instead of querying 'rates', query 'What are the current interest rates and loan repayment terms?').
   - ANSWERING RULE: Base your answer directly on the text returned by `search_knowledge_base`.
   - FALLBACK RULE: ONLY state "I don't have that specific information available right now. Is there anything else?" if `search_knowledge_base` returns "No relevant information found in knowledge base." or an empty response. NEVER invent or hallucinate policies.

4. HUMAN ESCALATION:
   - If the caller explicitly requests a human, say "I am transferring you to a human specialist now." and call the `transferCall` tool immediately.

5. BUSINESS ACTION TRIGGER:
   - Once all 4 qualification details are collected, call `submit_lead_qualification`.
"""

# 5. Create the Complete Assistant Payload
assistant_payload = {
    "name": "Business Loan Qualification Agent",
    "firstMessage": "Hello! Thank you for calling Apex Capital. Are you looking to apply for a business loan today?",
    "model": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "toolIds": [kb_tool_id, lead_tool_id, transfer_tool_id]
    },
    "voice": {
        "provider": "cartesia",
        "voiceId": "3b554273-4299-48b9-9aaf-eefd438e3941"
    },
    "transcriber": {
        "provider": "deepgram",
        "model": "nova-2-general"
    }
}

assistant_res = requests.post("https://api.vapi.ai/assistant", json=assistant_payload, headers=headers).json()
print("Assistant Created successfully!")
print("Assistant ID:", assistant_res["id"])