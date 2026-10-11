
import asyncio
import os
from agent_framework import Message
from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatClient

from careerpilot.models import CareerProfileExtraction

# Load the existing Azure OpenAI configuration
load_dotenv()

# Create the chat client using the existing environment variables
client = OpenAIChatClient(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Extract only career details explicitly stated by the user
async def main():
    # Supply a natural-language profile without a name
    user_text = (
        "I am a junior data analyst with 2 years of experience. "
        "I have a bachelor's degree in Computer Science. "
        "My skills include Python, SQL, Excel, and Power BI. "
        "I have a Microsoft Power BI Data Analyst certification. "
        "I work in financial services, and my goal is to transition "
        "into AI and machine learning."
    )

    
    # Send both framework messages as one sequence
    response = await client.get_response(
        messages=[
            Message(
                role="system",
                contents=[
                    (
                        "Extract career profile details from the user's text. "
                        "Use only information explicitly provided. "
                        "Return null for missing information. "
                        "Never infer or invent profile details."
                    )
                ],
            ),
            Message(
                role="user",
                contents=[user_text],
            ),
        ],
        options={"response_format": CareerProfileExtraction},
    )

    # Retrieve the validated extraction result
    profile = response.value
    print(profile.model_dump() if profile else "No structured profile returned")
    
# Run the asynchronous extraction test
asyncio.run(main())