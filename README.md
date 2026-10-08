# CareerPilot

**CareerPilot** is an AI-powered career transition and technology navigation agent.

It helps users understand technology career paths, evaluate their existing skills against a target career, identify skill gaps, and receive grounded career guidance.

---

## Current Capabilities

CareerPilot currently supports:

- Career knowledge questions
- Structured career lookups
- Career-fit prediction
- Multi-tool career analysis
- Out-of-scope request guardrails
- Tool-call limits
- Safe tool failure handling
- Human-review flags for significant skill gaps
- Agent execution tracing
- FastAPI endpoints
- Automated tests and evaluation cases

---

## Architecture

CareerPilot combines specialized tools behind an agent orchestration layer.

The current application exposes four main capabilities:

### `/ask`

Provides grounded answers using the CareerPilot knowledge base.

### `/predict`

Calculates a career-fit score based on the user's current skills and the required skills for a target career.

### `/lookup`

Retrieves structured information about supported technology career paths.

### `/agent`

Orchestrates the available tools to process a user's request and produce a structured career analysis.

---

## Agent Workflow

A typical career-fit analysis follows this workflow:

```text
                    User Request
                         |
                         v
                 CareerPilot Agent
                         |
              +----------+----------+
              |          |          |
              v          v          v
            lookup     predict      ask
              |          |          |
              +----------+----------+
                         |
                         v
              Structured Career Analysis
```

Depending on the user's request, the agent determines which tools are required and can combine multiple tools to generate the final response.

For example:

```text
"What skills do I need to become an AI Engineer,
and how well do my current skills match?"
                         |
                         v
                 CareerPilot Agent
                         |
              +----------+----------+
              |                     |
              v                     v
            lookup                predict
              |                     |
              +----------+----------+
                         |
                         v
              Structured Career Analysis
```

---

## Guardrails

CareerPilot includes several safeguards designed to make agent execution more predictable and reliable.

Current guardrails include:

- Scope validation for unrelated requests
- Maximum tool-call limits
- Safe handling of failed tools
- Input and output validation through Pydantic models
- Human-review flags for low-fit recommendations
- No hallucinated results when lookup information is unavailable
- Controlled access to available agent tools

These safeguards help ensure the agent produces structured and reliable responses while preventing unnecessary or unsupported tool execution.

---

## Project Structure

```text
careerpilot/
├── src/
│   └── careerpilot/
│       ├── agent.py
│       ├── api.py
│       ├── ask.py
│       ├── guardrails.py
│       ├── models.py
│       ├── predict.py
│       ├── state.py
│       ├── tools.py
│       └── tracing.py
├── tests/
│   ├── test_agent.py
│   └── test_api.py
├── note/
│   └── careerpilot_experiments.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

### Core Components

#### `agent.py`

Contains the main CareerPilot agent orchestration logic.

#### `api.py`

Defines the FastAPI application and API endpoints.

#### `ask.py`

Handles grounded career knowledge questions.

#### `guardrails.py`

Implements scope validation, tool-call limits, and other agent safety controls.

#### `models.py`

Contains Pydantic models used for request, response, and internal data validation.

#### `predict.py`

Implements career-fit prediction logic.

#### `state.py`

Manages structured agent state during execution.

#### `tools.py`

Defines and exposes the tools available to the CareerPilot agent.

#### `tracing.py`

Records agent execution and tool-call information for observability and debugging.

---

## Running the Project

### 1. Install Dependencies

Install project dependencies using `uv`:

```bash
uv sync
```

### 2. Run the Test Suite

Run all automated tests:

```bash
uv run pytest
```

### 3. Start the API

Start the FastAPI development server:

```bash
uv run uvicorn careerpilot.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive FastAPI documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

CareerPilot currently exposes the following primary endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/ask` | Answer grounded career knowledge questions |
| `/predict` | Calculate career-fit scores |
| `/lookup` | Retrieve structured career information |
| `/agent` | Execute multi-tool career analysis |

---

## Testing

The project contains automated tests covering both agent behavior and API functionality.

Current test coverage includes:

- Agent tool selection
- Multi-tool orchestration
- Out-of-scope requests
- Tool failure handling
- Tool-call guardrails
- Human-review requirements
- API endpoints
- Agent API integration

The current test suite contains:

```text
17 passing tests
```

Run the tests with:

```bash
uv run pytest
```

For concise test output:

```bash
uv run pytest -q
```

---

## Technology Stack

CareerPilot currently uses:

- Python
- FastAPI
- Uvicorn
- Pydantic
- uv
- Pytest
- Jupyter
- Git

---

## Agent Design

CareerPilot is designed around a tool-using agent architecture.

Instead of relying on a single model call to handle every request, the agent can select and combine specialized tools depending on the user's intent.

```text
                       User
                         |
                         v
                  CareerPilot Agent
                         |
              Intent + Scope Analysis
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
        lookup         predict          ask
          |              |              |
          +--------------+--------------+
                         |
                         v
                 Structured Result
                         |
                         v
                  Human Review
                   if required
```

This architecture separates:

- Agent reasoning
- Tool execution
- Career knowledge retrieval
- Career-fit prediction
- Validation
- Guardrails
- Execution tracing

This separation makes the system easier to test, extend, and monitor.

---

## Human Review

CareerPilot can flag recommendations for human review when a significant skill gap is detected.

This is particularly useful for cases where the system should avoid presenting an automated recommendation as definitive.

The general workflow is:

```text
Career Analysis
      |
      v
Career-Fit Evaluation
      |
      v
Significant Skill Gap?
      |
   +--+--+
   |     |
  No    Yes
   |     |
   v     v
Result  Human Review Flag
```

The human-review mechanism provides an additional control layer for higher-impact recommendations.

---

## Error Handling

CareerPilot is designed to fail safely when a tool cannot successfully complete its task.

Examples include:

- Unsupported career lookups
- Tool execution failures
- Requests outside the supported scope
- Invalid request data
- Tool-call limit violations

Rather than fabricating unavailable information, the agent returns controlled responses when required data cannot be retrieved.

---

## Observability and Tracing

Agent execution tracing is included to make CareerPilot's behavior easier to inspect and debug.

Tracing can capture information such as:

```text
User Request
     |
     v
Agent Decision
     |
     v
Selected Tool
     |
     v
Tool Result
     |
     v
Next Agent Decision
     |
     v
Final Response
```

This provides visibility into how the agent processed a request and which tools were involved in producing the result.

---

## Project Direction

CareerPilot is intended to evolve toward a production-style tool-using AI agent.

Planned areas of development include:

- Microsoft Agent Framework
- Structured agent state
- Human-in-the-loop workflows
- Improved observability and tracing
- Automated evaluation pipelines
- Retrieval-Augmented Generation (RAG)
- PostgreSQL
- pgvector
- Docker
- Production API deployment

---

## Planned Architecture

The target architecture extends the current agent into a more complete AI system:

```text
                         User
                           |
                           v
                    CareerPilot API
                           |
                           v
                    CareerPilot Agent
                           |
              +------------+------------+
              |            |            |
              v            v            v
            Lookup       Predict       RAG
              |            |            |
              |            |      +-----+------+
              |            |      |            |
              |            |      v            v
              |            |   PostgreSQL   pgvector
              |            |      |            |
              +------------+------+------------+
                           |
                           v
                    Agent State
                           |
                           v
                     Guardrails
                           |
                           v
                    Human Review
                    when required
                           |
                           v
                    Final Response
                           |
                           v
                 Tracing & Evaluation
```

---

## Project Goals

The long-term goal of CareerPilot is to provide a practical AI career navigation system that can:

1. Understand a user's career goal.
2. Retrieve structured information about the target career.
3. Evaluate the user's current skills.
4. Calculate career-fit indicators.
5. Identify missing or underdeveloped skills.
6. Retrieve grounded career knowledge.
7. Recommend relevant development areas.
8. Combine multiple tools within a single agent workflow.
9. Escalate significant recommendations for human review where appropriate.
10. Trace and evaluate agent behavior.
11. Provide its capabilities through a production-ready API.

---

## Current Status

The current implementation provides the foundation for a broader production-style AI agent.

```text
Career Knowledge Questions       ✅
Structured Career Lookup         ✅
Career-Fit Prediction            ✅
Multi-Tool Orchestration         ✅
Scope Guardrails                 ✅
Tool-Call Limits                 ✅
Safe Tool Failure Handling       ✅
Human-Review Flags               ✅
Agent Execution Tracing          ✅
FastAPI Integration              ✅
Automated Tests                  ✅
Evaluation Cases                 ✅
Microsoft Agent Framework        🔄 Planned
RAG Integration                  🔄 Planned
PostgreSQL + pgvector            🔄 Planned
Docker                           🔄 Planned
Production Deployment            🔄 Planned
```

CareerPilot will continue to evolve from this foundation toward a more robust, observable, grounded, and production-ready career navigation agent.