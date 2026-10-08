import os
from careerpilot.tracing import AgentTracer
from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from careerpilot.framework_tools import framework_career_lookup
from careerpilot.guardrails import (
    scope_guardrail,
    tool_failure_guardrail,
    tool_not_found_guardrail,
)


# Load environment variables from the local .env file
load_dotenv()


# Create the Azure OpenAI chat client using environment variables
client = OpenAIChatClient(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Invoke the framework tool using its structured arguments
import asyncio


# Run the asynchronous framework tool
async def main():
    # Create the CareerPilot agent with the lookup tool
    agent = Agent(
        client=client,
        instructions=(
            "You are CareerPilot, an AI career navigation agent. "
            "Use the career_lookup tool when you need structured "
            "information about technology careers."
        ),
        tools=[framework_career_lookup],
    )

    # Create a tracer for this agent run
    tracer = AgentTracer()
    
    # Invoke the CareerPilot lookup wrapper with structured arguments
    """result = await framework_career_lookup.invoke(
        arguments={
            "request": {
                "lookup_type": "career",
                "query": "AI Engineer",
            }
        }
    )

    # Extract and display the actual text returned by the framework tool
    print(result[0].text)"""

    # Define the user's request for this agent run
    user_request = (
        "Look up the CareerPilot information for the AI Engineer career. "
        "Tell me the required skills, preferred skills, and related roles."
    )

    # Check the request against the existing CareerPilot scope guardrail
    guardrail_response = scope_guardrail(user_request)

    # Stop before calling the model when the request is out of scope
    if guardrail_response is not None:
        print("Guardrail:", guardrail_response)
        return

    # Run the Agent Framework agent for an allowed request
    response = await agent.run(user_request)

    # Check whether any tool returned an explicit failure
    for message in response.messages:
        for content in message.contents:
            # Inspect tool results before accepting the agent response
            if content.type == "function_result":
                tool_result = content.result

                # Stop unsafe fallback responses after a tool failure
                guardrail_response = tool_failure_guardrail(tool_result)

                if guardrail_response is not None:
                    print("Guardrail:", guardrail_response)
                    return

                # Stop the agent from inventing information when a lookup finds nothing
                guardrail_response = tool_not_found_guardrail(tool_result)

                if guardrail_response is not None:
                    print("Guardrail:", guardrail_response)
                    return

    # Trace each message produced during the agent run
    for message in response.messages:
        tracer.log_framework_message(message)

# Start the async experiment
asyncio.run(main())


# Verify the configured tool invocation limit
print("Tool name:", framework_career_lookup.name)
print("Maximum invocations:", framework_career_lookup.max_invocations)