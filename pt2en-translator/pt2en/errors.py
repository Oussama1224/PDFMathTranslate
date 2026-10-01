"""Exception types with user-facing messages."""


class PipelineError(Exception):
    """An error that should be shown to the user as-is."""

    def __init__(self, message: str, *, stage: str = "", detail: str = ""):
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.detail = detail


class InvalidDocumentError(PipelineError):
    pass


class ProviderConfigurationError(PipelineError):
    pass


class JobCancelled(Exception):
    """Raised inside the pipeline when the user cancels a job."""
