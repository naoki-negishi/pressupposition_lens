from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    """A prompt for training LREs and LRCs"""

    text: str
    answer: str
    subject: str
    subject_name: str = ""  # If not provided, will be set to subject
    object_name: str = ""  # If not provided, will be set to answer

    def __post_init__(self) -> None:
        if self.subject_name == "":
            object.__setattr__(self, "subject_name", self.subject)
        if self.object_name == "":
            object.__setattr__(self, "object_name", self.answer)


@dataclass(frozen=True, slots=True)
class PromptNLI:
    text: str  # concatation of premise and hypothesis
    answer: str  # trigger
    premise: str
    premise_sent: str  # If not provided, will be set to premise
    trigger_type: str  # If not provided, will be set to answer

    def __post_init__(self) -> None:
        if self.premise_sent == "":
            object.__setattr__(self, "premise_sent", self.premise)
        if self.trigger_type == "":
            object.__setattr__(self, "trigger_type", self.answer)
