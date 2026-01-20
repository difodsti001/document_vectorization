"""
FastAPI Backend - Sistema de Vectorización de Documentos
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from settings import settings, validate_settings
from services import (
    load_and_save_document,
    validate_document_structure,
    extract_normalize_and_hash,
    chunk_text_semantic,
    init_vectorization_service
)


# ==================== LIFESPAN EVENT ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando aplicación...")
    
    # Crear directorio templates si no existe
    Path("frontend").mkdir(exist_ok=True)
    
    validate_settings()
    init_vectorization_service()
    print("✅ VectorizationService inicializado")
    yield
    print("👋 Cerrando aplicación...")

# ==================== APP ====================
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="frontend")

UPLOAD_DIR = settings.UPLOAD_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

batch_processing_status: Dict[str, Dict[str, Any]] = {}

# ==================== INTERFAZ WEB ====================

@app.get("/", response_class=HTMLResponse, tags=["Interfaz"])
async def home(request: Request):
    """
    🌐 Interfaz web principal del sistema de vectorización
    
    Accede aquí para:
    - Seleccionar colecciones
    - Cargar documentos
    - Vectorizar archivos
    - Buscar por similitud semántica
    """
    return templates.TemplateResponse("interfaz.html", {"request": request})

# ==================== MODELOS ====================

class FileValidationInfo(BaseModel):
    """Información de validación de UN archivo"""
    file_id: str
    filename: str
    total_pages: int
    total_images: int
    total_tables: int
    has_non_textual_content: bool
    status: str
    error_message: Optional[str] = None
    is_duplicate: bool = False
    duplicate_chunks: int = 0


class BatchValidationResponse(BaseModel):
    """Respuesta de validación para múltiples archivos"""
    batch_id: str
    total_files: int
    validated_files: int
    failed_files: int
    files: List[FileValidationInfo]
    message: str


class BatchProgressResponse(BaseModel):
    """Progreso de procesamiento del batch"""
    batch_id: str
    total_files: int
    completed_files: int
    failed_files: int
    current_file: Optional[str] = None
    overall_progress: int
    files_status: List[Dict[str, Any]]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


# ==================== ENDPOINTS ====================

@app.get("/health", tags=["Sistema"])
async def health_check():
    """Health check del servicio"""
    return {
        "status": "healthy",
        "service": "document-vectorization",
        "version": settings.API_VERSION,
        "endpoints": {
            "interfaz_web": "http://localhost:8100/",
            "api_docs": "http://localhost:8100/docs"
        }
    }


@app.post("/api/upload-batch", response_model=BatchValidationResponse, tags=["Vectorización"])
async def upload_batch(collection_name: str = Query(..., description="Colección destino"),
    files: List[UploadFile] = File(...)
    ):
    """
    Endpoint 1: Upload y validación de MÚLTIPLES documentos
    Incluye verificación de duplicados por nombre de archivo
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos")

    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Máximo 20 archivos por lote")

    batch_id = str(uuid.uuid4())
    validated_count = 0
    failed_count = 0
    files_info = []
    saved_files_paths = {}

    from services import check_duplicate_by_filename

    for file in files:
        file_id = str(uuid.uuid4())

        try:
            if not file.filename.lower().endswith((".pdf", ".docx")):
                files_info.append(FileValidationInfo(
                    file_id=file_id,
                    filename=file.filename,
                    total_pages=0,
                    total_images=0,
                    total_tables=0,
                    has_non_textual_content=False,
                    status="error",
                    error_message="Formato no soportado"
                ))
                failed_count += 1
                continue

            file_path = UPLOAD_DIR / file.filename
            if file_path.exists():
                file_path.unlink()
            with file_path.open("wb") as buffer:
                buffer.write(await file.read())

            saved_files_paths[file_id] = str(file_path)

            validation = await validate_document_structure(file_path)

            has_non_textual = (
                validation["total_images"] > 0 or
                validation["total_tables"] > 0
            )


            duplicate_check = check_duplicate_by_filename(file.filename, collection_name)
            is_duplicate = duplicate_check["exists"]
            duplicate_chunks = duplicate_check["total_chunks"]

            files_info.append(FileValidationInfo(
                file_id=file_id,
                filename=file.filename,
                total_pages=validation["total_pages"],
                total_images=validation["total_images"],
                total_tables=validation["total_tables"],
                has_non_textual_content=has_non_textual,
                status="validated",
                is_duplicate=is_duplicate, 
                duplicate_chunks=duplicate_chunks  
            ))

            validated_count += 1

        except Exception as e:
            files_info.append(FileValidationInfo(
                file_id=file_id,
                filename=file.filename,
                total_pages=0,
                total_images=0,
                total_tables=0,
                has_non_textual_content=False,
                status="error",
                error_message=str(e)
            ))
            failed_count += 1

    batch_files = {}
    for f in files_info:
        if f.status == "validated":
            batch_files[f.file_id] = {
                "file_path": saved_files_paths[f.file_id],
                "filename": f.filename,
                "validation": {
                    "total_pages": f.total_pages,
                    "total_images": f.total_images,
                    "total_tables": f.total_tables
                },
                "status": "validated",
                "progress": 0,
                "error": None,
                "is_duplicate": f.is_duplicate, 
                "duplicate_chunks": f.duplicate_chunks
            }

    batch_processing_status[batch_id] = {
        "files": batch_files,
        "overall_status": "validated",
        "total_files": len(files),
        "completed_files": 0,
        "failed_files": failed_count,
        "collection_name": collection_name  
    }

    message = f"✅ {validated_count} archivo(s) validado(s)"
    if failed_count > 0:
        message += f" | ⚠️ {failed_count} archivo(s) con error"

    return BatchValidationResponse(
        batch_id=batch_id,
        total_files=len(files),
        validated_files=validated_count,
        failed_files=failed_count,
        files=files_info,
        message=message
    )

@app.post("/api/confirm-duplicates/{batch_id}", tags=["Vectorización"])
async def confirm_duplicate_replacement(
    batch_id: str,
    files_to_replace: List[str] = Query(..., description="Lista de file_ids a reemplazar")
):
    """
    Endpoint para confirmar qué duplicados se quieren reemplazar
    El frontend debe llamar esto ANTES de vectorizar
    """
    if batch_id not in batch_processing_status:
        raise HTTPException(status_code=404, detail="Batch no encontrado")

    batch = batch_processing_status[batch_id]
    collection_name = batch["collection_name"]

    from services import delete_document_from_collection

    deleted_summary = []

    for file_id in files_to_replace:
        if file_id not in batch["files"]:
            continue

        file_info = batch["files"][file_id]
        filename = file_info["filename"]

        result = delete_document_from_collection(filename, collection_name)

        deleted_summary.append({
            "filename": filename,
            "success": result["success"],
            "deleted_chunks": result.get("deleted_chunks", 0)
        })

        file_info["duplicate_deleted"] = True

    return {
        "success": True,
        "deleted_documents": deleted_summary,
        "message": f"✅ {len(deleted_summary)} documento(s) antiguo(s) eliminado(s)"
    }


@app.post("/api/vectorize-batch/{batch_id}", tags=["Vectorización"])
async def start_batch_vectorization(
    batch_id: str,
    background_tasks: BackgroundTasks,
    collection_name: str = Query(..., description="Nombre de la colección de destino")
):
    """
    Endpoint 2: Inicia vectorización de TODOS los archivos del batch en una colección específica
    """
    if batch_id not in batch_processing_status:
        raise HTTPException(status_code=404, detail="Batch no encontrado")

    batch = batch_processing_status[batch_id]

    if batch["overall_status"] != "validated":
        raise HTTPException(status_code=400, detail="Batch no validado")

    batch["collection_name"] = collection_name
    batch["overall_status"] = "processing"

    background_tasks.add_task(process_batch_vectorization, batch_id)

    return {
        "message": "Vectorización de batch iniciada",
        "batch_id": batch_id,
        "collection_name": collection_name,
        "total_files": batch["total_files"]
    }


async def process_batch_vectorization(batch_id: str):
    """
    Procesa TODOS los archivos del batch secuencialmente
    Ya NO verifica duplicados aquí (se hizo antes)
    """
    batch = batch_processing_status[batch_id]
    collection_name = batch["collection_name"]
    completed = 0
    failed = 0

    service = init_vectorization_service()

    for  file_id,file_info in batch["files"].items():
        try:
            batch["current_file"] = file_info["filename"]
            file_info["status"] = "processing"

            file_path = file_info["file_path"]
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

            print(f"📄 Procesando: {file_path}")

            file_info["progress"] = 25
            file_info["current_step"] = "Extrayendo y normalizando..."

            extracted = await extract_normalize_and_hash(file_path)
            normalized_text = extracted["normalized_text"]
            document_hash = extracted["document_hash"]

            file_info["progress"] = 50
            file_info["current_step"] = "Segmentando documento..."

            chunks = await chunk_text_semantic(normalized_text)

            file_info["progress"] = 75
            file_info["current_step"] = "Generando embeddings..."

            embeddings = await service.embed([c["text"] for c in chunks])

            metadata = {
                "document_hash": document_hash,
                "filename": file_info["filename"],
                "format": Path(file_path).suffix.replace(".", ""),
                "total_pages": file_info["validation"]["total_pages"],
                "total_chunks": len(chunks)
            }

            result = await service.store_vectors(
                collection_name=collection_name,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )

            file_info["status"] = "completed"
            file_info["progress"] = 100
            file_info["current_step"] = "✅ Completado"
            file_info["result"] = result
            completed += 1

            Path(file_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"❌ Error procesando {file_info['filename']}: {e}")
            file_info["status"] = "failed"
            file_info["progress"] = 0
            file_info["current_step"] = f"Error: {e}"
            failed += 1

        batch["completed_files"] = completed
        batch["failed_files"] = failed

    batch["overall_status"] = "completed"
    batch["current_file"] = None


@app.get("/api/batch-progress/{batch_id}", response_model=BatchProgressResponse, tags=["Vectorización"])
async def get_batch_progress(batch_id: str):
    """
    Endpoint 3: Obtiene el progreso del batch completo
    """
    if batch_id not in batch_processing_status:
        raise HTTPException(status_code=404, detail="Batch no encontrado")

    batch = batch_processing_status[batch_id]

    files_status = [
        {
            "file_id": file_id,
            "filename": info["filename"],
            "status": info["status"],
            "progress": info.get("progress", 0),
            "current_step": info.get("current_step", "Esperando..."),
            "error": info.get("error")
        }
        for file_id, info in batch["files"].items()
    ]

    total_files = len(batch["files"])
    completed = batch.get("completed_files", 0)
    overall_progress = int((completed / total_files) * 100) if total_files > 0 else 0

    return BatchProgressResponse(
        batch_id=batch_id,
        total_files=total_files,
        completed_files=completed,
        failed_files=batch.get("failed_files", 0),
        current_file=batch.get("current_file"),
        overall_progress=overall_progress,
        files_status=files_status
    )


@app.delete("/api/batch/{batch_id}", tags=["Vectorización"])
async def delete_batch(batch_id: str):
    """
    Endpoint 4: Limpia un batch completado
    """
    if batch_id in batch_processing_status:
        batch = batch_processing_status[batch_id]

        for file_info in batch["files"].values():
            file_path = file_info.get("file_path")
            if file_path:
                Path(file_path).unlink(missing_ok=True)

        del batch_processing_status[batch_id]
        return {"message": "Batch eliminado"}

    raise HTTPException(status_code=404, detail="Batch no encontrado")


@app.post("/api/search", tags=["Búsqueda"])
async def search_similar_documents(
    request: SearchRequest,
    collection_name: str = Query(..., description="Colección a consultar")
):
    """
    Endpoint 5: Búsqueda por embeddings
    """
    service = init_vectorization_service()

    query_vector = await service.embed([request.query], is_query=True)

    results = service._client.search(
        collection_name=collection_name,
        query_vector=query_vector[0],
        limit=request.top_k
    )

    return {
        "query": request.query,
        "collection": collection_name,
        "results": [
            {
                "score": round(hit.score, 4),
                "text": hit.payload.get("text"),
                "filename": hit.payload.get("filename"),
                "chunk": hit.payload.get("chunk")
            }
            for hit in results
        ]
    }

@app.post("/api/cleanup-uploads", tags=["Mantenimiento"])
async def cleanup_uploads():
    """
    Endpoint 6: Mantenimiento, limpia archivos temporales huérfanos
    """
    try:
        deleted_count = 0
        upload_dir = Path("./uploads")
        
        if not upload_dir.exists():
            return {"message": "Directorio uploads no existe", "deleted": 0}
        
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_age = datetime.datetime.now().timestamp()
                
                if file_age > 3600: 
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        print(f"🗑️ Eliminado: {file_path.name} (antigüedad: {file_age/60:.1f} min)")
                    except Exception as e:
                        print(f"⚠️ No se pudo eliminar {file_path.name}: {e}")
        
        return {
            "success": True,
            "deleted_files": deleted_count,
            "message": f"✅ {deleted_count} archivo(s) temporal(es) eliminado(s)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en limpieza: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🌐 SISTEMA DE VECTORIZACIÓN DOCUMENTAL")
    print("="*70)
    print(f"📍 Interfaz Web: http://{settings.API_HOST}:{settings.API_PORT}/vectorización")
    print(f"📖 Documentación: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"🔧 API Colecciones: http://localhost:9000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        log_level="info"
    )