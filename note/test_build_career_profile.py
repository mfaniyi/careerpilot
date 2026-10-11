
# Import the agent and the partial profile model
from careerpilot.agent import CareerPilotAgent
from careerpilot.models import CareerProfileExtraction


# Create an agent instance
agent = CareerPilotAgent()

# Create a complete profile using explicitly supplied test data
extracted = CareerProfileExtraction(
    name="Test User",
    current_role="Junior Data Analyst",
    years_of_experience=2,
    education=["Bachelor's degree in Computer Science"],
    skills=["Python", "SQL", "Excel", "Power BI"],
    certifications=["Microsoft Power BI Data Analyst"],
    industry="Financial services",
    career_goal="Transition into AI and machine learning",
)

# Convert the extraction into the validated CareerProfile model
profile = agent._build_career_profile(extracted)

# Confirm that the returned model is the expected type
assert profile.name == "Test User"
assert profile.years_of_experience == 2
assert "Python" in profile.skills

# Confirm that the helper rejects incomplete profiles
incomplete = CareerProfileExtraction(
    current_role="Junior Data Analyst",
    years_of_experience=2,
)

try:
    agent._build_career_profile(incomplete)
except ValueError as error:
    # Verify that the error identifies the missing required fields
    assert "name" in str(error)
    print("Incomplete profile rejected safely.")
else:
    # Fail the test if incomplete data is accepted
    raise AssertionError("An incomplete profile should have been rejected.")

# Report successful completion
print("Career profile validation test passed.")