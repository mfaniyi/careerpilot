from careerpilot.ask import AskRequest, ask
from careerpilot.predict import CareerFitRequest, predict_career_fit
from careerpilot.state import AgentState
from careerpilot.tools import lookup
from careerpilot.models import CareerAnalysis, CareerProfile, LookupRequest
from careerpilot.tracing import AgentTracer
from careerpilot.guardrails import (
    requires_human_review,
    scope_guardrail,
    tool_call_guardrail,
)

# Main CareerPilot agent
class CareerPilotAgent:

    def __init__(self) -> None:
        # Create a tracer for this agent instance
        self.tracer = AgentTracer()

    def run(
        self,
        request: str,
        profile: CareerProfile | None = None,
    ) -> AgentState:
        """Process a user request using CareerPilot tools."""

        # Create state for this agent run
        state = AgentState(user_request=request)

        # Record the incoming request
        self.tracer.log(f"Agent received request: {request}")

        # Check whether the request is within CareerPilot's scope
        refusal = scope_guardrail(request)

        if refusal:
            self.tracer.log("Guardrail triggered: request is out of scope")
            state.final_response = refusal
            return state

        # Decide what to do based on simple intent rules
        request_lower = request.lower()

        # Handle general career knowledge questions
        if "what does" in request_lower or "what is" in request_lower:
            self.tracer.log("Selected tool: ask")

            # Stop if the tool-call limit has been reached
            guardrail = tool_call_guardrail(state.tool_call_count)

            if guardrail:
                state.final_response = guardrail
                return state

            # Call the knowledge tool
            result = ask(AskRequest(question=request))

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

        # Handle career-fit questions using the combined analysis workflow
        if (
            profile is not None
            and ("fit" in request_lower or "suitable" in request_lower)
        ):
            self.tracer.log("Selected workflow: career analysis")

            # Stop if the tool-call limit has been reached
            guardrail = tool_call_guardrail(state.tool_call_count)

            if guardrail:
                state.final_response = guardrail
                return state

            # Identify the requested career
            target_role = None

            known_roles = [
                "Data Analyst",
                "Data Engineer",
                "AI Engineer",
                "Business Analyst",
            ]

            for role in known_roles:
                if role.lower() in request_lower:
                    target_role = role
                    break

            # Use Data Analyst when no specific career is mentioned
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

                # Log when human review is required
                if state.human_review_required:
                    self.tracer.log("Human review required for this recommendation")


            except RuntimeError as error:
                # Log the controlled failure without exposing raw exceptions
                self.tracer.log("Career analysis stopped safely")

                # Return known safe messages while hiding unexpected errors
                safe_messages = {
                    "The career-fit prediction service is currently unavailable.",
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

            # Log the analysis result
            self.tracer.log(
                f"Career analysis result: {analysis.model_dump()}"
            )

            # Store the structured final response
            state.final_response = analysis.model_dump_json(
                indent=2
            )

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

            result = lookup(
                LookupRequest(
                    lookup_type="career",
                    query=target_role,
                )
            )

            # Record the tool execution
            state.add_tool_step(
                tool="lookup",
                arguments={
                    "lookup_type": "career",
                    "query": target_role,
                },
                result=result.model_dump_json() if result else "Not found",
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

            # Log the successful lookup
            self.tracer.log(
                f"Lookup result: {result.model_dump()}"
            )

            # Store the verified result as the final answer
            state.final_response = result.model_dump_json(indent=2)

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

    career = lookup(
        LookupRequest(
            lookup_type="career",
            query=target_role,
        )
    )

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
        prediction = predict_career_fit(
            CareerFitRequest(
                profile=profile,
                target_role=target_role,
            )
        )

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

    explanation = ask(
        AskRequest(
            question=f"What does a {career.role} do?"
        )
    )

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

    # Combine all tool results into one structured analysis
    return CareerAnalysis(
        target_role=career.role,
        career_information=career,
        fit_score=prediction.score,
        matched_skills=prediction.matched_skills,
        missing_skills=prediction.missing_skills,
        career_explanation=explanation,
        recommendation=prediction.recommendation,
    )