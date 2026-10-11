
# Import the agent and the partial profile model
from careerpilot.agent import CareerPilotAgent
from careerpilot.models import CareerProfileExtraction


# Create an agent instance for the helper test
agent = CareerPilotAgent()

# Supply a profile with a name and certifications missing
profile = CareerProfileExtraction(
    current_role="Junior Data Analyst",
    years_of_experience=2,
    education=["Bachelor's degree in Computer Science"],
    skills=["Python", "SQL"],
    industry="Financial services",
    career_goal="Transition into AI and machine learning",
)

# Identify which required fields still need to be collected
missing = agent._get_missing_profile_fields(profile)

# Verify that only the omitted fields are reported
assert missing == ["name", "certifications"], missing

# Display the result and confirm the test passed
print("Missing fields:", missing)
print("Missing profile fields test passed.")