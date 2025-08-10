import asyncio
import os
from openai import AsyncOpenAI
from agents import Runner, Agent, GuardrailFunctionOutput, InputGuardrail, OutputGuardrail, set_tracing_disabled
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Disabled tracing to avoid overhead or API Key requirments
set_tracing_disabled = True


#Configure Gemini API client
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

#Define Context for user account detail
class BankContext(BaseModel):
    account_number: Optional[str] = None
    balance: float = 1000  

#Input Guardrail to validate account number
class AccountInput(BaseModel):
    is_valid: bool
    reason: str

async def account_input_guardrail(ctx, agent, input_data):
    account_num = input_data.lower().split("Account Number")[-1].strip() if "Account Number" in input_data else None
    is_valid = account_num and account_num.isdigit() and len(account_num) == 16
    return GuardrailFunctionOutput(
        output_info=AccountInput(is_valid=is_valid, reason="Account number is valid" if is_valid else "Invalid or missing account number"),
        tripwire_triggered=not is_valid
    )

# output guardrail to ensure polite and accurate response
class PoliteOutput(BaseModel):
    is_polite: bool
    reason: str

async def polite_output_guardrail(ctx, agent, output_data):
    is_polite = "sorry" not in output_data.lower() and "error" not in output_data.lower()
    return GuardrailFunctionOutput(
        output_info=PoliteOutput(is_polite=is_polite, reason="Response is polite and professional" if is_polite else "Response contains negative words"),
        tripwire_triggered=not is_polite
    )

#Define Agents
balance_agent = Agent(
    name="BalanceAgent",
    instructions="Your provide account balance information. Use the account number from the context and return the balance in a friendly meanner.",
    model= "gemini-2.5-flash",
    output_guardrails=[OutputGuardrail(guardrail_function=polite_output_guardrail)]
)

transaction_agent = Agent(
    name="TransactionAgent",
    instructions= "You process transaction request. Confirm the transaction amount and update the balance in the contet. Ensure the amount is positive and insufficient funds are available.",
    model="gemini-2.5-flash",
    output_guardrails=[OutputGuardrail(guardrail_function=polite_output_guardrail)]
)

triage_agent = Agent(
    name= " TriageAgent",
    instructions="You are a bank Customer service agent. Route queries to the BalanceAgent for balance inquiries or TransactionAgent for transaction requests. Ask for account number if not provided.",
    model="gemini-2.5-flash",
    handoffs= [balance_agent, transaction_agent],
    input_guardrails=[InputGuardrail(guardrail_function=account_input_guardrail)],
)

async def main():
    queries = [
        "What's my Balance? Account Number 1234567890123456",
        "Transfer $200. Account number is 9876543210987654",
        "Invalid query. Account number abc", 
        "What's my balance? Account number 1234567890123456",
    ]

    context = BankContext(account_number=None, balance=1000)

    for query in queries:
        try:
            result = await Runner.run(triage_agent, query, context=context)
            print(f"Query: {query}")
            print(f"Response: {result.final_output}")
            print(f"Update Balance: {context.balance}\n")
        except Exception as e:
            print(f"Query: {query}")
            print(f"Error: {str(e)}\n")

if __name__ == "__main__":
    asyncio.run(main())

