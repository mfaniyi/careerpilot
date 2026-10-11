# Import libraries for JSON handling, environment variables, and HTTP requests
import json
import os
import httpx
import asyncio
from dotenv import load_dotenv
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient
from careerpilot.models import CareerProfileExtraction
from careerpilot.predict import CareerFitPrediction
from careerpilot.state import AgentState
from careerpilot.models import CareerAnalysis, CareerPath, CareerProfile
from careerpilot.tracing import AgentTracer
from careerpilot.guardrails import (
    requires_human_review,
    scope_guardrail,
    tool_call_guardrail,
)

# Allow the API address to be configured for different environments
API_BASE_URL = os.getenv(
    "CAREERPILOT_API_BASE_URL",
    "http://127.0.0.1:8000",
)
# Main CareerPilot agent
class CareerPilotAgent:

    def __init__(self) -> None:
        # Create a tracer for this agent instance
        self.tracer = AgentTracer()


    
    async def _extract_profile(self, user_text: str) -> CareerProfileExtraction:
        """Extract explicitly stated career details into a validated model."""

        # Load Azure OpenAI settings from the existing environment configuration
        load_dotenv()

        # Create the chat client using the project's existing Azure OpenAI credentials
        client = OpenAIChatClient(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

        # Await the asynchronous API call without starting another event loop
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
            # Require the model's response to follow our Pydantic schema
            options={"response_format": CareerProfileExtraction},
        )

        # Retrieve the structured extraction result
        profile = response.value

        # Fail safely if the model returns no structured result
        if profile is None:
            raise ValueError("No structured career profile was returned.")

        return profile



    def _get_missing_profile_fields(
        self,
        extracted: CareerProfileExtraction,
    ) -> list[str]:
        """Return the required profile fields that are still missing."""

        # List the fields required by the complete CareerProfile model
        required_fields = [
            "name",
            "current_role",
            "years_of_experience",
            "education",
            "skills",
            "certifications",
            "industry",
            "career_goal",
        ]

        # Identify fields that are absent, empty, or contain no usable values
        missing_fields = []
        for field in required_fields:
            value = getattr(extracted, field)

            # Treat None, blank strings, and empty lists as missing
            if value is None or value == "" or value == []:
                missing_fields.append(field)

        return missing_fields



    def _build_career_profile(
        self,
        extracted: CareerProfileExtraction,
    ) -> CareerProfile:
        """Convert a complete extraction into a validated career profile."""

        # Find any required fields that still need user input
        missing_fields = self._get_missing_profile_fields(extracted)

        # Prevent incomplete profile data from entering career analysis
        if missing_fields:
            raise ValueError(
                f"Missing required profile fields: {', '.join(missing_fields)}"
            )

        # Validate all extracted values against the complete profile schema
        return CareerProfile.model_validate(extracted.model_dump())

    def _merge_pending_profile(
        self,
        existing: CareerProfileExtraction,
        updates: CareerProfileExtraction,
    ) -> CareerProfileExtraction:
        """Merge new profile details while preserving previously collected skills."""

        # Start with the profile details collected from earlier messages.
        merged_data = existing.model_dump()
        update_data = updates.model_dump()

        # Merge skill lists so new details do not erase previously supplied skills.
        existing_skills = merged_data.get("skills") or []
        updated_skills = update_data.get("skills") or []

        # Preserve skill spelling while avoiding case-insensitive duplicates.
        combined_skills = []
        seen_skills = set()

        for skill in existing_skills + updated_skills:
            normalized_skill = skill.strip().casefold()

            # Ignore empty values and duplicate skills.
            if normalized_skill and normalized_skill not in seen_skills:
                combined_skills.append(skill.strip())
                seen_skills.add(normalized_skill)

        merged_data["skills"] = combined_skills

        # Update other fields only when the new extraction provides a value.
        for field, value in update_data.items():
            if field == "skills":
                continue

            if value is not None:
                merged_data[field] = value

        # Validate the merged result using the existing profile schema.
        return CareerProfileExtraction.model_validate(merged_data)


    def _post_to_api(self, endpoint: str, payload: dict) -> dict:
        """Send a POST request to a CareerPilot API endpoint."""

        # Send the request with a timeout to prevent indefinite waiting
        response = httpx.post(
            f"{API_BASE_URL}/{endpoint}",
            json=payload,
            timeout=10.0,
        )

        # Raise an error if the API returns an unsuccessful status code
        response.raise_for_status()

        # Return the response body as a Python dictionary
        return response.json()


    def run(
        self,
        request: str,
        profile: CareerProfile | None = None,
        pending_profile: CareerProfileExtraction | None = None,
    ) -> AgentState:
        """Process a user request using CareerPilot tools."""

        # Create state for this agent run
        state = AgentState(user_request=request)

        # Record the incoming request
        self.tracer.log(f"Agent received request: {request}")

        # Check whether the request is within CareerPilot's scope
        # Allow follow-up answers when collecting an incomplete career profile
        refusal = None if pending_profile is not None else scope_guardrail(request)

        if refusal:
            self.tracer.log("Guardrail triggered: request is out of scope")
            state.final_response = refusal
            return state

        # Decide what to do based on simple intent rules
        request_lower = request.lower()

        # Recognize career-fit and career-recommendation requests
        career_fit_triggers = (
            "fit",
            "suitable",
            "suitability",
            "assess my",
            "assess my suitability",
            "assess my fit",
            "how well do i fit",
            "recommend a career",
            "recommend careers",
            "recommend the skills",
            "which career",
            "which tech career",
            "what career",
            "career recommendation",
            "best career for me",
            "skills i should learn",
        )

        # Continue profile collection when an earlier request left fields missing
        if pending_profile is not None or any(
            trigger in request_lower for trigger in career_fit_triggers
        ):

            
            # Extract profile details when the caller has not supplied a structured profile
            
            # Build a profile from the current message when no complete profile was supplied
            if profile is None:
                try:
                    # Extract details stated in the latest user message
                    extracted = asyncio.run(self._extract_profile(request))

                    # Merge the new details with any profile from an earlier turn
                    if pending_profile is not None:
                        extracted = self._merge_pending_profile(
                            existing=pending_profile,
                            updates=extracted,
                        )

                    # Identify any required fields that are still missing
                    missing_fields = self._get_missing_profile_fields(extracted)

                    # Ask for missing information before running analysis
                    if missing_fields:
                        field_questions = {
                            "name": "your name",
                            "current_role": "your current role",
                            "years_of_experience": "your years of experience",
                            "education": "your educational background",
                            "skills": "your skills",
                            "certifications": "your certifications, or 'none'",
                            "industry": "your industry",
                            "career_goal": "your career goal",
                        }

                        # Convert missing fields into a readable follow-up question
                        questions = [
                            field_questions[field]
                            for field in missing_fields
                        ]

                        # Preserve the partial profile so the client can send it back
                        state.pending_profile = extracted

                        state.final_response = (
                            "I need a little more information before "
                            "I can recommend a suitable career. Please provide: "
                            + "; ".join(questions)
                        )

                        # Log field names only, not the user's profile values
                        self.tracer.log(
                            f"Profile extraction requires fields: {missing_fields}"
                        )
                        return state

                    # Validate the completed profile before career analysis
                    profile = self._build_career_profile(extracted)

                except Exception:
                    # Avoid exposing internal errors or credentials to the user
                    self.tracer.log("Profile extraction failed safely")
                    state.final_response = (
                        "I couldn't reliably extract your career background. "
                        "Please describe your current role, years of experience, "
                        "education, skills, certifications, industry, and career goal."
                    )
                    return state


            self.tracer.log("Selected workflow: career analysis")

            # Stop if the tool-call limit has been reached
            guardrail = tool_call_guardrail(state.tool_call_count)

            if guardrail:
                state.final_response = guardrail
                return state

            # Identify the requested career, prioritizing the user's career goal.
            target_role = None


            # Define supported careers and their common keywords.
            goal_role_mapping = {
                "AI Engineer": [
                    "artificial intelligence",
                    "machine learning",
                    "ai engineer",
                    "ai engineering",
                    "generative ai",
                ],
                "Data Engineer": [
                    "data engineering",
                    "data engineer",
                    "data pipelines",
                ],
                "Data Analyst": [
                    "data analysis",
                    "data analyst",
                    "data analytics",
                ],
                "Business Analyst": [
                    "business analysis",
                    "business analyst",
                ],
            }

            # Identify an explicitly requested target role in the question.
            target_role = None

            # Match phrases that indicate the role being assessed.
            explicit_role_phrases = [
                ("AI Engineer", ["for ai engineer", "for an ai engineer", "as an ai engineer"]),
                ("Data Engineer", ["for data engineer", "for a data engineer", "as a data engineer"]),
                ("Data Analyst", ["for data analyst", "for a data analyst", "as a data analyst"]),
                ("Business Analyst", ["for business analyst", "for a business analyst", "as a business analyst"]),
            ]

            # Give the role explicitly requested for assessment the highest priority.
            for role, phrases in explicit_role_phrases:
                if any(phrase in request_lower for phrase in phrases):
                    target_role = role
                    break

            # If no explicit target was found, use the extracted career goal.
            if target_role is None and profile is not None and profile.career_goal:
                career_goal_lower = profile.career_goal.lower()

                # Match the career goal to a supported career.
                for role, keywords in goal_role_mapping.items():
                    if any(keyword in career_goal_lower for keyword in keywords):
                        target_role = role
                        break

            # Use Data Analyst only when no target can be identified.
            if target_role is None:
                target_role = "Data Analyst"

            # Run the career analysis workflow safely
            try:
                analysis = analyze_career(
                    target_role=target_role,
                    profile=profile,
                    state=state,
                    tracer=self.tracer,
                )

                # Check whether the recommendation requires human review
                state.human_review_required = requires_human_review(
                    fit_score=analysis.fit_score or 0,
                    recommendation=analysis.recommendation or "",
                )


                # Store the assessment when a human review is required.
                if state.human_review_required:
                    # Queue the assessment and generate its review identifier.
                    state.queue_for_review(analysis.model_dump())
                else:
                    # Mark assessments that do not require review.
                    state.review_status = "not_required"
                    state.review_id = None
                    state.pending_review = None

                # Log when human review is required
                if state.human_review_required:
                    self.tracer.log("Human review required for this recommendation")


            except RuntimeError as error:
                # Log the controlled failure without exposing raw exceptions
                self.tracer.log("Career analysis stopped safely")

                # Return known safe messages while hiding unexpected errors
                safe_messages = {
                    "The career-fit prediction service is currently unavailable.",
                    "The career lookup service is currently unavailable.",
                }

                if str(error).startswith("Career path not found:"):
                    state.final_response = (
                        "Career path not found in the CareerPilot knowledge base."
                    )
                elif str(error) in safe_messages:
                    state.final_response = str(error)
                else:
                    state.final_response = (
                        "I couldn't complete the career analysis because a "
                        "required service encountered an error. Please try again later."
                    )

                return state

            # Log when the assessment must wait for human review.
            if state.human_review_required:
                self.tracer.log(
                    "Assessment generated and marked pending human review"
                )

                # Preserve the assessment JSON for existing consumers.
                state.final_response = analysis.model_dump_json(indent=2)
                return state

            # Log the analysis result when no review is required.
            self.tracer.log(
                f"Career analysis result: {analysis.model_dump()}"
            )

            # Store the structured final response.
            state.final_response = analysis.model_dump_json(indent=2)
            return state

        # Handle general career knowledge and advice questions
        knowledge_triggers = (
            "what does",
            "what is",
            "what are",
            "how do",
            "how can",
            "how to",
            "explain",
            "describe",
            "tell me about",
            "which career",
            "what career",
            "career path",
            "suitable for",
            "interested in",
            "enjoys working",
            "skills needed",
        )

        if any(trigger in request_lower for trigger in knowledge_triggers):
            self.tracer.log("Selected tool: ask")

            # Stop if the tool-call limit has been reached
            guardrail = tool_call_guardrail(state.tool_call_count)

            if guardrail:
                state.final_response = guardrail
                return state


            # Call the knowledge endpoint and handle service failures safely
            try:
                response = self._post_to_api("ask", {"question": request})

                # Extract the answer returned by the API
                result = response["answer"]
            except Exception:
                # Log a safe message without exposing internal service details
                self.tracer.log("Knowledge service failed")
                state.final_response = (
                    "The knowledge service is currently unavailable. Please try again later."
                )
                return state

            # Record the tool execution
            state.add_tool_step(
                tool="ask",
                arguments={"question": request},
                result=result,
            )

            self.tracer.log(f"Tool result: {result}")

            # Store the final answer
            state.final_response = result
            return state

        # Handle explicit career lookups
        if "lookup" in request_lower or "look up" in request_lower:
            self.tracer.log("Selected tool: lookup")

            # Stop if the tool-call limit has been reached
            guardrail = tool_call_guardrail(state.tool_call_count)

            if guardrail:
                state.final_response = guardrail
                return state

            # Remove common lookup phrases to extract the requested career
            target_role = (
                request_lower
                .replace("please lookup", "")
                .replace("please look up", "")
                .replace("lookup", "")
                .replace("look up", "")
                .strip()
            )

            # Use AI Engineer as a fallback when no career was specified
            if not target_role:
                target_role = "AI Engineer"

            # Call the career lookup endpoint and handle service failures safely
            try:
                response = self._post_to_api(
                    "lookup",
                    {
                        "lookup_type": "career",
                        "query": target_role,
                    },
                )

                # Extract the career record when a match is found
                result = response.get("career") if response.get("found") else None
            except Exception:
                # Log a safe message without exposing internal service details
                self.tracer.log("Career lookup service failed")
                state.final_response = (
                    "The career lookup service is currently unavailable. Please try again later."
                )
                return state


            # Record the tool execution
            state.add_tool_step(
                tool="lookup",
                arguments={
                    "lookup_type": "career",
                    "query": target_role,
                },
                # Store the career record as JSON when a match is found
                result=json.dumps(result) if result else "Not found",
            )

            # Handle a missing lookup result safely
            if result is None:
                self.tracer.log("Lookup failed: career not found")

                # Never invent information when the tool has no result
                state.final_response = (
                    f"I couldn't find verified information about '{target_role}'. "
                    "I won't invent details that are not in the CareerPilot knowledge base."
                )

                return state

            # Log the career record returned by the API
            self.tracer.log(f"Lookup result: {result}")

            # Store the verified API result as formatted JSON
            state.final_response = json.dumps(result, indent=2)

            return state

        # Safely handle requests the current agent cannot process
        self.tracer.log("No matching tool strategy found")

        state.final_response = (
            "I understand the request, but I don't yet have a tool "
            "strategy for this type of question."
        )

        return state


def analyze_career(
    target_role: str,
    profile: CareerProfile,
    state: AgentState,
    tracer: AgentTracer,
) -> CareerAnalysis:
    """Combine lookup, predict, and ask into one career analysis."""

    # Check the tool-call limit before starting the workflow
    guardrail = tool_call_guardrail(state.tool_call_count)

    if guardrail:
        raise RuntimeError(guardrail)

    # Log the lookup tool and its arguments
    tracer.log(
        "Career analysis step 1: calling lookup",
        details={
            "lookup_type": "career",
            "query": target_role,
        },
    )


    # Request career information from the FastAPI lookup endpoint
    try:
        response = httpx.post(
            f"{API_BASE_URL}/lookup",
            json={
                "lookup_type": "career",
                "query": target_role,
            },
            timeout=10.0,
        )

        # Raise an error if the lookup API returns an unsuccessful status
        response.raise_for_status()

        # Extract and validate the career record from the API response
        lookup_data = response.json()
        career = (
            CareerPath.model_validate(lookup_data["career"])
            if lookup_data.get("found") and lookup_data.get("career")
            else None
        )
    except Exception:
        # Avoid exposing internal service or database details
        tracer.log("Career lookup service failed")
        raise RuntimeError(
            "The career lookup service is currently unavailable."
        ) from None

    # Record the lookup execution
    state.add_tool_step(
        tool="lookup",
        arguments={
            "lookup_type": "career",
            "query": target_role,
        },
        result=career.model_dump_json() if career else "Not found",
    )

    # Stop safely if the career does not exist
    if career is None:
        raise RuntimeError(f"Career path not found: {target_role}")

    # Check the tool-call limit before calling the prediction tool
    guardrail = tool_call_guardrail(state.tool_call_count)

    if guardrail:
        raise RuntimeError(guardrail)

    # Log the prediction tool and its arguments
    tracer.log(
        "Career analysis step 2: calling predict",
        details={
            "target_role": target_role,
        },
    )
    
    # Try to calculate the candidate's career fit
    try:

        # Send the candidate profile to the prediction API
        response = httpx.post(
            f"{API_BASE_URL}/predict",
            json={
                "profile": profile.model_dump(),
                "target_role": target_role,
            },
            timeout=10.0,
        )

        # Raise an error if the prediction API returns an unsuccessful status
        response.raise_for_status()

        # Validate the JSON response using the actual prediction model
        prediction = CareerFitPrediction.model_validate(response.json())

    except Exception as error:
        # Log the tool failure without exposing internal details
        tracer.log("Predict tool failed")

        # Record the failed tool execution
        state.add_tool_step(
            tool="predict",
            arguments={
                "target_role": target_role,
            },
            result=f"Tool error: {type(error).__name__}",
        )

        # Stop the workflow safely
        raise RuntimeError(
            "The career-fit prediction service is currently unavailable."
        )

    # Record the prediction execution
    state.add_tool_step(
        tool="predict",
        arguments={
            "target_role": target_role,
        },
        result=prediction.model_dump_json(),
    )

    # Check the tool-call limit before calling the ask tool
    guardrail = tool_call_guardrail(state.tool_call_count)

    if guardrail:
        raise RuntimeError(guardrail)

    # Log the knowledge tool and its arguments
    tracer.log(
        "Career analysis step 3: calling ask",
        details={
            "question": f"What does a {career.role} do?",
        },
    )



    # Ask the knowledge endpoint to explain the selected career
    try:
        response = httpx.post(
            f"{API_BASE_URL}/ask",
            json={"question": f"What does a {career.role} do?"},
            timeout=10.0,
        )

        # Raise an error if the knowledge API returns an unsuccessful status
        response.raise_for_status()

        # Extract the explanation from the API response
        explanation = response.json()["answer"]
    except Exception:
        # Log a safe message without exposing internal service details
        tracer.log("Career explanation service failed")
        raise RuntimeError(
            "The career explanation service is currently unavailable."
        ) from None

    # Record the knowledge lookup execution
    state.add_tool_step(
        tool="ask",
        arguments={
            "question": f"What does a {career.role} do?",
        },
        result=explanation,
    )


    # Log the knowledge tool result for auditability
    tracer.log(
        "Ask tool completed",
        details={
            "tool": "ask",
            "result": explanation,
        },
    )

    # Combine all tool results into one structured analysis.
    return CareerAnalysis(
        target_role=career.role,
        career_information=career,
        fit_score=prediction.score,
        matched_skills=prediction.matched_skills,
        missing_skills=prediction.missing_skills,
        matched_preferred_skills=prediction.matched_preferred_skills,
        career_explanation=explanation,
        recommendation=prediction.recommendation,
    )