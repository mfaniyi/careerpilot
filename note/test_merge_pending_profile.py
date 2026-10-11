
# Import the agent and the partial profile model
from careerpilot.agent import CareerPilotAgent
from careerpilot.models import CareerProfileExtraction


# Create an agent instance
agent = CareerPilotAgent()

# Represent the background extracted from the first message
existing = CareerProfileExtraction(
    current_role="Junior Data Analyst",
    years_of_experience=2,
    education=["Bachelor's degree in Computer Science"],
    skills=["Python", "SQL", "Excel", "Power BI"],
    certifications=["Microsoft Power BI Data Analyst certification"],
    industry="Financial services",
    career_goal="Transition into AI and machine learning",
)

# Represent the user's follow-up answer
updates = CareerProfileExtraction(name="Michael")

# Merge the new answer into the original extracted profile
merged = agent._merge_pending_profile(existing, updates)

# Verify that the missing name was added
assert merged.name == "Michael"

# Verify that previously collected details were preserved
assert merged.current_role == "Junior Data Analyst"
assert merged.years_of_experience == 2
assert "Python" in merged.skills
assert merged.certifications == [
    "Microsoft Power BI Data Analyst certification"
]

# Display the result and confirm the test passed
print("Merged profile:", merged.model_dump())
print("Pending profile merge test passed.")