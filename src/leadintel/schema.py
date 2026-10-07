"""The lead record: the single validated input shape for training, scoring and the UI.

Field meanings follow the UCI Bank Marketing documentation (dataset 222, ``bank-full``).
``duration`` is deliberately *not* part of a lead: it is only known after the final call,
when the outcome is already decided (see docs/DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md).
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Job = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
Marital = Literal["divorced", "married", "single"]
Education = Literal["primary", "secondary", "tertiary", "unknown"]
YesNo = Literal["no", "yes"]
Contact = Literal["cellular", "telephone", "unknown"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
PrevOutcome = Literal["failure", "other", "success", "unknown"]

MONTHS: tuple[str, ...] = Month.__args__  # calendar order, used for display and encoding

# Categorical levels, used to build encoders with a fixed, data-independent vocabulary.
CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "job": Job.__args__,
    "marital": Marital.__args__,
    "education": Education.__args__,
    "default": YesNo.__args__,
    "housing": YesNo.__args__,
    "loan": YesNo.__args__,
    "contact": Contact.__args__,
    "month": MONTHS,
    "poutcome": PrevOutcome.__args__,
}


class Lead(BaseModel):
    """One lead as seen at decision time (before the next contact is made)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Customer profile
    age: int = Field(ge=18, le=100, description="Age in years")
    job: Job
    marital: Marital
    education: Education
    default: YesNo = Field(description="Has credit in default?")
    balance: int = Field(ge=-20_000, le=200_000, description="Average yearly balance (EUR)")
    housing: YesNo = Field(description="Has a housing loan?")
    loan: YesNo = Field(description="Has a personal loan?")

    # Campaign context
    contact: Contact = Field(description="Channel of the current campaign's contacts")
    day: int = Field(ge=1, le=31, description="Day of month of the latest contact")
    month: Month = Field(description="Month of the latest contact")

    # Interaction history
    campaign: int = Field(ge=1, le=100, description="Contacts made in this campaign")
    pdays: int = Field(
        ge=-1, le=999, description="Days since a previous-campaign contact (-1 = never)"
    )
    previous: int = Field(ge=0, le=300, description="Contacts before this campaign")
    poutcome: PrevOutcome = Field(description="Outcome of the previous campaign")

    @model_validator(mode="after")
    def _history_is_consistent(self) -> Self:
        # These two rules hold for every one of the 45,211 UCI rows; a record that breaks
        # them describes a history that cannot happen, so we refuse to score it.
        if (self.previous == 0) != (self.pdays == -1):
            raise ValueError(
                "Inconsistent history: 'contacts before this campaign' is 0 exactly when "
                "'days since previous contact' is -1 (never contacted)."
            )
        if self.previous == 0 and self.poutcome != "unknown":
            raise ValueError(
                "Inconsistent history: a lead never contacted before cannot have a "
                "previous-campaign outcome. Set previous outcome to 'unknown'."
            )
        return self
