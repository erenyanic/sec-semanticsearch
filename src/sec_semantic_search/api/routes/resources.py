"""
GPU resource management endpoints.

Provides ``GET /api/resources/gpu`` to check model status and
``DELETE /api/resources/gpu`` to explicitly unload the embedding
model and free VRAM.

``gpu_status`` stays async: it reads flags and never touches the model.
``gpu_unload`` is plain ``def`` because unloading waits for any in-flight
encode to release the embedder lock.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from sec_semantic_search.api.dependencies import get_embedder, get_task_manager, verify_admin_key
from sec_semantic_search.api.schemas import ErrorResponse, GPUStatusResponse, GPUUnloadResponse
from sec_semantic_search.api.tasks import TaskManager
from sec_semantic_search.core import audit_log, get_logger
from sec_semantic_search.pipeline import EmbeddingGenerator

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/gpu",
    response_model=GPUStatusResponse,
    summary="GPU / model status",
)
async def gpu_status(
    embedder: EmbeddingGenerator = Depends(get_embedder),
) -> GPUStatusResponse:
    """
    Check whether the embedding model is loaded, which device it is
    on, and approximate VRAM usage.

    This endpoint never triggers model loading.
    """
    return GPUStatusResponse(
        model_loaded=embedder.is_loaded,
        device=embedder.device if embedder.is_loaded else None,
        model_name=embedder.model_name,
        approximate_vram_mb=embedder.approximate_vram_mb,
    )


@router.delete(
    "/gpu",
    response_model=GPUUnloadResponse,
    responses={403: {"model": ErrorResponse}},
    summary="Unload embedding model",
    dependencies=[Depends(verify_admin_key)],
)
def gpu_unload(
    request: Request,
    embedder: EmbeddingGenerator = Depends(get_embedder),
    task_manager: TaskManager = Depends(get_task_manager),
) -> GPUUnloadResponse:
    """
    Unload the embedding model and free GPU memory.

    Returns 409 if an ingestion task is currently pending or running.
    The model will reload automatically on the next search or ingest.
    """
    if task_manager.has_active_task():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Conflict",
                "message": "Cannot unload model while tasks are active.",
                "hint": "Wait for running tasks to complete or cancel them first.",
            },
        )

    if not embedder.is_loaded:
        logger.info("GPU unload requested but model is not loaded.")
        return GPUUnloadResponse(status="already_unloaded")

    embedder.unload()

    client_ip = request.client.host if request.client else "unknown"
    audit_log(
        "gpu_unload",
        client_ip=client_ip,
        endpoint="DELETE /api/resources/gpu",
    )
    return GPUUnloadResponse(status="unloaded")
