
# Import asyncio to run the asynchronous extraction method
import asyncio

# Import the CareerPilot agent
from careerpilot.agent import CareerPilotAgent


# Test natural-language extraction without running the full workflow
async def main():
    # Create the agent instance
    agent = CareerPilotAgent()

    # Supply a profile that deliberately omits the user's name
    user_text = (
        "I am a junior data analyst with 2 years of experience. "
        "I have a bachelor's degree in Computer Science. "
        "My skills include Python, SQL, Excel, and Power BI. "
        "I have a Microsoft Power BI Data Analyst certification. "
        "I work in financial services, and my goal is to transition "
        "into AI and machine learning."
    )

    # Extract the profile using the new agent method
    profile = await agent._extract_profile(user_text)

    # Display the structured result for inspection
    print(profile.model_dump())

    # Confirm that the missing name was not fabricated
    assert profile.name is None, "The model should not invent a name."

    # Confirm that the stated experience was extracted correctly
    assert profile.years_of_experience == 2

    # Confirm that the stated skills were extracted
    assert "Python" in profile.skills
    assert "SQL" in profile.skills

    # Confirm that the extraction identified the career goal
    assert profile.career_goal is not None
    assert "AI" in profile.career_goal or "machine learning" in profile.career_goal.lower()

    # Confirm that the test passed
    print("\nProfile extraction test passed.")


# Run the asynchronous test
if __name__ == "__main__":
    asyncio.run(main())