from pydantic import BaseModel, Field
from careerpilot.models import CareerProfileExtraction
from uuid import uuid4

# Records one tool execution performed by the agent
class ToolStep(BaseModel):
    tool: str
    arguments: dict
    result: str


# Stores the agent's state throughout a request
class AgentState(BaseModel):
    user_request: str
    tool_steps: list[ToolStep] = Field(default_factory=list)
    tool_call_count: int = 0
    human_review_required: bool = False
    review_status: str = "not_required"
    review_id: str | None = None
    pending_review: dict | None = None
    final_response: str | None = None
    pending_profile: CareerProfileExtraction | None = None

    # Record a completed tool call
    def add_tool_step(
        self,
        tool: str,
        arguments: dict,
        result: str,
    ) -> None:
        self.tool_steps.append(
            ToolStep(
                tool=tool,
                arguments=arguments,
                result=result,
            )
        )

        # Keep track of the number of tool calls
        self.tool_call_count += 1


    
    # Queue an assessment for human review and assign a unique ID.
    def queue_for_review(self, assessment: dict) -> None:
        # Mark the assessment as requiring a human decision.
        self.human_review_required = True
        self.review_status = "pending"

        # Generate an identifier for tracking the review.
        self.review_id = str(uuid4())

        # Preserve the assessment until a reviewer decides.
        self.pending_review = assessment