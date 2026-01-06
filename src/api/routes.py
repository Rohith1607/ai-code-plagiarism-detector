from fastapi import APIRouter
from src.api.schemas import AnalyzeRequest, AnalyzeResponse
from src.pipeline.orchestrator import AnalysisPipeline

router = APIRouter(prefix="/analyze", tags=["analysis"])

pipeline = AnalysisPipeline()


@router.post("/", response_model=AnalyzeResponse)
def analyze_code(request: AnalyzeRequest):

    result = pipeline.run(
        code=request.code,
        language=request.language
    )

    return AnalyzeResponse(**result)
