"""API and artifact contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .banding import RiskBand


class PredictRequest(BaseModel):
    """One loan applicant.

    Only income, credit amount, and date of birth are required. Everything else
    is optional because thin-file borrowers, by definition, have sparse records —
    and a missing external score is signal the model uses rather than an error.

    CODE_GENDER is deliberately absent: sex is a prohibited basis for credit
    decisions under ECOA / Regulation B.
    """

    model_config = ConfigDict(extra="forbid")

    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_CREDIT: float = Field(gt=0)
    DAYS_BIRTH: float = Field(lt=0, description="Days before application; negative")

    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)
    AMT_ANNUITY: float | None = Field(default=None, gt=0)
    AMT_GOODS_PRICE: float | None = Field(default=None, gt=0)
    DAYS_EMPLOYED: float | None = Field(
        default=None, description="Negative day count, or 365243 meaning not employed"
    )
    CNT_CHILDREN: int | None = Field(default=None, ge=0)
    CNT_FAM_MEMBERS: float | None = Field(default=None, ge=1)

    FLAG_OWN_CAR: str | None = None
    FLAG_OWN_REALTY: str | None = None
    NAME_CONTRACT_TYPE: str | None = None
    NAME_INCOME_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_HOUSING_TYPE: str | None = None
    OCCUPATION_TYPE: str | None = None


class Reason(BaseModel):
    """One principal factor increasing an applicant's risk.

    `code` is stable and machine-readable; `description` is the plain-language
    text a creditor can put in front of an applicant. Contribution magnitudes are
    deliberately not exposed: they are meaningless to an applicant and would make
    the model straightforward to reverse-engineer.
    """

    code: str
    description: str


class PredictResponse(BaseModel):
    # 'model_' is a protected Pydantic namespace; disabling it lets us name the
    # field model_version, which is what the field actually is.
    model_config = ConfigDict(protected_namespaces=())

    probability: float = Field(ge=0, le=1)
    risk_band: RiskBand
    model_version: str
    reasons: list[Reason] = Field(
        default_factory=list,
        description=(
            "Principal factors increasing this applicant's risk, most significant "
            "first, at most four. Where a caller takes adverse action on this "
            "score, these are the specific principal reasons ECOA / Regulation B "
            "1002.9 requires. Empty when no factor pushed the score upward."
        ),
    )


class GroupRate(BaseModel):
    """One group's outcome rate. `n` travels with the rate so a reader can judge
    whether a ratio near the threshold is signal or noise."""

    group: str
    adverse_rate: float = Field(ge=0, le=1)
    n: int = Field(ge=0)


class AttributeFairness(BaseModel):
    """One protected attribute's adverse impact ratio, or the reason there isn't
    one. Exactly one of the two is ever set: an attribute that could not be
    measured has established nothing, and must never read as having passed."""

    attribute: str
    adverse_impact_ratio: float | None = Field(default=None, ge=0, le=1)
    unmeasured_reason: str | None = None
    groups: list[GroupRate] = Field(default_factory=list)

    @model_validator(mode="after")
    def exactly_one_outcome(self) -> AttributeFairness:
        measured = self.adverse_impact_ratio is not None
        unmeasured = self.unmeasured_reason is not None
        if measured == unmeasured:
            raise ValueError(
                f"attribute {self.attribute!r} must be either measured "
                "(adverse_impact_ratio) or unmeasured (unmeasured_reason), never "
                "both and never neither"
            )
        return self


class FairnessReport(BaseModel):
    """Disparate impact across protected attributes, measured on the validation
    split at training time.

    band_low_max records the policy the measurement was taken under. Risk-band
    thresholds are business policy that changes without retraining, so a stored
    ratio can go stale; recording the threshold makes that discoverable rather
    than silent. When band policy moves, fairness must be re-measured.
    """

    adverse_definition: str
    band_low_max: float = Field(ge=0, le=1)
    min_group_size: int = Field(ge=1)
    attributes: list[AttributeFairness]


class ModelMetadata(BaseModel):
    """Sidecar written next to model.json. The feature_order field is the gate
    that prevents a model being served against a mismatched transform."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    trained_at: str
    dataset_sha256: str
    n_train_rows: int
    feature_order: list[str]
    metrics: dict[str, float]
    xgboost_version: str
    provenance: Literal["fixture", "production"]
