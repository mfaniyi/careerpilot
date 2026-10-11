
# Import the CareerPilot agent
from careerpilot.agent import CareerPilotAgent


# Test profile extraction through the main agent workflow
def main():
    # Create an agent instance
    agent = CareerPilotAgent()

    # Provide a career background that deliberately omits the user's name
    request = (
        "I am a junior data analyst with 2 years of experience. "
        "I have a bachelor's degree in Computer Science. "
        "My skills include Python, SQL, Excel, and Power BI. "
        "I have a Microsoft Power BI Data Analyst certification. "
        "I work in financial services, and my career goal is to transition "
        "into AI and machine learning. Which tech career is best for me?"
    )

    # Run the main workflow, which should extract the profile first
    state = agent.run(request=request)

    # Display the follow-up response for inspection
    print("Agent response:")
    print(state.final_response)

    # Confirm that the agent asks for the missing name
    assert "your name" in state.final_response.lower(), (
        "The agent should ask for the missing name."
    )

    # Confirm that analysis tools have not run prematurely
    assert state.tool_call_count == 0, (
        "No analysis tools should run while required details are missing."
    )

    # Confirm successful completion
    print("\nProfile follow-up test passed.")


# Run the test
if __name__ == "__main__":
    main()