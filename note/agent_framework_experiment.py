import asyncio
import os

from dotenv import load_dotenv
from agent_framework import Agent, Message
from agent_framework.openai import OpenAIChatClient

from careerpilot.tracing import AgentTracer
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

    # Create a session managed by the agent for approval state persistence
    session = agent.create_session()

    # Run the agent using the approval-enabled session
    response = await agent.run(
        user_request,
        session=session,
    )

    # Retrieve approval requests through the framework's public API
    approval_request = response.user_input_requests[0]

    # Approve the requested tool call using its occurrence ID
    approval_response = approval_request.to_function_approval_response(
        approved=True
    )

    # Wrap the approval response in an explicit user message
    approval_message = Message(
        role="user",
        contents=[approval_response],
    )

    approved_response = await agent.run(approval_message, session=session)

    # Display the result returned by the approved tool call
    for message in approved_response.messages:
        for content in message.contents:
            if content.type == "function_result":
                print("Tool result:", content.result)

    # Check whether any tool returned an explicit failure
    for message in approved_response.messages:
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
    for message in approved_response.messages:
        tracer.log_framework_message(message)

# Run the asynchronous agent experiment
asyncio.run(main())
