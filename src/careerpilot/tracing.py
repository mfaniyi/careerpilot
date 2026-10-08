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