"""Optional product routes; failures do not join the Q&A loading path."""
from pathlib import Path
from typing import Any, Callable
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.database import SourceFileError
from app.services import exploration_sources, source_exploration


class ExplorationResponse(BaseModel):
    identity: dict[str, Any]
    data_version: str
    algorithm: str
    coverage: dict[str, Any]
    status: str | None = None
    candidates: list[dict[str, Any]] | None = None
    pairs: list[dict[str, Any]] | None = None
    pair: dict[str, Any] | None = None
    total: int | None = None
    all_pairs: int | None = None
    offset: int | None = None
    limit: int | None = None
    normalized_error: str | None = None
    boundary: str | None = None
    traceback: dict[str, Any] | None = None
    unsupported: list[dict[str, Any]] | None = None


class ErrorQuery(BaseModel):
    revision: str = Field(min_length=1,max_length=128)
    error: str = Field(min_length=1,max_length=4096)
    offset: int = Field(default=0,ge=0,le=100000)
    limit: int = Field(default=20,ge=1,le=50)


class MaterialRequest(BaseModel):
    revision: str = Field(min_length=1,max_length=128)
    local_path: str = Field(min_length=1,max_length=2048)


def create_router(database: Callable[[],Any], runtime: Path) -> APIRouter:
    router=APIRouter(prefix='/api/projects/{project_id}/exploration',tags=['source exploration'])

    def run(action: Callable[[],Any]) -> Any:
        try: return action()
        except SourceFileError as exc:
            raise HTTPException(exc.status_code,detail={'code':exc.code,'message':'当前版本源码材料不可用，请刷新项目或检查同版本材料。','retryable':False}) from exc
        except (OSError,ValueError,RecursionError) as exc:
            raise HTTPException(422,detail={'code':'exploration_failed','message':'源码探索暂时无法完成；问答与其他页面仍可使用。','retryable':True}) from exc

    def load(project_id: str,revision: str) -> dict[str,Any]:
        return source_exploration.analysis(database(),runtime,project_id,revision)

    @router.get('',response_model=ExplorationResponse,response_model_exclude_none=True)
    def overview(project_id: str,revision: str=Query(...,min_length=1,max_length=128)) -> Any:
        return run(lambda:source_exploration.envelope(load(project_id,revision)))

    @router.post('/errors',response_model=ExplorationResponse,response_model_exclude_none=True)
    def errors(project_id: str,request: ErrorQuery) -> Any:
        return run(lambda:source_exploration.error_query(load(project_id,request.revision),request.error,request.offset,request.limit))

    @router.get('/comparisons',response_model=ExplorationResponse,response_model_exclude_none=True)
    def pairs(project_id: str,revision: str=Query(...,min_length=1,max_length=128),q: str=Query('',max_length=200),
              status: str=Query('',pattern='^(|supported|partial)$'),representative: bool=True,
              offset: int=Query(0,ge=0,le=100000),limit: int=Query(20,ge=1,le=50)) -> Any:
        return run(lambda:source_exploration.comparisons(load(project_id,revision),q,status,representative,offset,limit))

    @router.get('/comparisons/{pair_id}',response_model=ExplorationResponse,response_model_exclude_none=True)
    def detail(project_id: str,pair_id: str,revision: str=Query(...,min_length=1,max_length=128)) -> Any:
        return run(lambda:source_exploration.comparison(load(project_id,revision),pair_id))

    @router.post('/materials',response_model=ExplorationResponse,response_model_exclude_none=True)
    def materials(project_id: str,request: MaterialRequest) -> Any:
        def prepare() -> Any:
            exploration_sources.supplement(database(),runtime,project_id,request.revision,request.local_path)
            return source_exploration.envelope(load(project_id,request.revision))
        return run(prepare)

    return router
