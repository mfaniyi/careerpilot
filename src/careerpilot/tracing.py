from datetime import datetime


# Store and display agent execution events
class AgentTracer:

    def __init__(self) -> None:
        # Keep a record of all events during the agent run
        self.events: list[str] = []

    def log(self, message: str, details: dict | None = None) -> None:
        """Record and display an agent event with optional details."""

        # Create a timestamp for the event
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Start with the main event message
        event = f"[{timestamp}] {message}"

        # Add structured details when provided
        if details is not None:
            event += f" | Details: {details}"

        # Store the event
        self.events.append(event)

        # Display the event immediately
        print(event)


        # Trace messages produced by Microsoft Agent Framework
    def log_framework_message(self, message) -> None:
        """Record important Agent Framework execution events."""

        # Record the message role
        self.log(
            "Framework message received",
            {"role": message.role},
        )

        # Inspect each content item in the message
        for content in message.contents:

            # Record function/tool calls
            if content.type == "function_call":
                self.log(
                    "Tool selected",
                    {
                        "tool": content.name,
                        "arguments": content.arguments,
                    },
                )

            # Record tool results
            elif content.type == "function_result":
                self.log(
                    "Tool result received",
                    {
                        "result": content.result,
                        "status": content.additional_properties.get("status"),
                    },
                )

            # Record normal text responses
            elif content.type == "text":
                self.log(
                    "Agent response generated",
                    {"text": content.text},
                )