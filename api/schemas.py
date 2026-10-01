from typing import Dict, List, Literal
from pydantic import BaseModel, Field

# Label mapping for model outputs
LABEL_MAPPING: Dict[int, str] = {
    0: "Neutral",
    1: "Beneficial",
    2: "Negative",
}


class StudentFeatures(BaseModel):
    Age: int = Field(..., ge=10, le=100, description="Student age in years", example=20)
    Gender: Literal["Female", "Male", "Non-Binary", "Prefer not to say"] = Field(
        ..., description="Gender identity", example="Male"
    )
    Academic_Level: Literal["Undergraduate", "High School", "Postgraduate"] = Field(
        ..., description="Current educational stage", example="Undergraduate"
    )
    Primary_Platform: Literal[
        "Instagram", "TikTok", "YouTube", "Snapchat", "LinkedIn", "Reddit", "X (Twitter)"
    ] = Field(..., description="Most frequently used platform", example="Instagram")
    Daily_Usage_Hours: float = Field(
        ..., ge=0.0, le=24.0, description="Average daily hours on social media", example=6.5
    )
    Weekend_Extra_Hours: float = Field(
        ..., ge=0.0, le=24.0, description="Additional hours spent on weekends", example=2.0
    )
    Device_Type: Literal["Smartphone", "Tablet", "Laptop/PC"] = Field(
        ..., description="Primary device used", example="Smartphone"
    )
    Sleep_Duration_Hours: float = Field(
        ..., ge=0.0, le=24.0, description="Average sleep per night in hours", example=6.0
    )
    Sleep_Quality_Score: int = Field(
        ..., ge=1, le=5, description="Self-reported sleep quality (1 to 5)", example=5
    )
    Late_Night_Usage: int = Field(
        ...,
        ge=0,
        le=1,
        description="Late night social media usage flag (0 for No, 1 for Yes)",
        example=1,
    )
    Social_Comparison_Frequency: Literal[
        "Never", "Rarely", "Sometimes", "Frequently", "Always"
    ] = Field(
        ..., description="Frequency of comparing oneself to others online", example="Frequently"
    )
    Perceived_Stress_Score: float = Field(
        ..., ge=0.0, le=50.0, description="Stress assessment score", example=8.0
    )
    Mental_Health_Index: int = Field(
        ..., ge=0, le=100, description="Mental health wellness index", example=45
    )
    Academic_Performance_GPA: float = Field(
        ..., ge=0.0, le=4.0, description="Current GPA on 4.0 scale", example=2.7
    )


class SinglePredictionResponse(BaseModel):
    prediction_code: int = Field(..., description="Predicted class integer (0, 1, or 2)")
    prediction_label: str = Field(..., description="Human-readable impact label (Neutral, Beneficial, Negative)")
    probabilities: Dict[str, float] = Field(
        ..., description="Class probability distribution"
    )


class BatchPredictionRequest(BaseModel):
    students: List[StudentFeatures] = Field(..., description="List of student data records")


class BatchPredictionResponse(BaseModel):
    total_samples: int
    predictions: List[SinglePredictionResponse]
