# 📚 Sistema de Vectorización Documental

Un sistema completo de procesamiento y vectorización de documentos educativos que permite almacenar, organizar y buscar contenido mediante embeddings semánticos.

## 🎯 Características Principales

- **Carga Múltiple de Documentos**: Soporta la carga simultánea de hasta 20 archivos PDF y DOCX
- **Validación Inteligente**: Detecta estructura del documento, imágenes, tablas y contenido no textual
- **Detección de Duplicados**: Identifica documentos duplicados antes de la vectorización
- **Vectorización Semántica**: Utiliza modelos BGE para generar embeddings de alta calidad
- **Búsqueda Semántica**: Busca documentos por similitud usando vectores
- **Gestión de Colecciones**: Integración con sistema externo de colecciones
- **Interfaz Moderna**: Panel web intuitivo con diseño responsive basado en MINEDU

## 📋 Requisitos Previos

- **Python** >= 3.9
- **Qdrant** corriendo en `localhost:6333`
- API de Colecciones en `localhost:9000`

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd vectorizacion_final
```
### 2. Crear entorno virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```
### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crear archivo .env en la raíz del proyecto:
```bash
    # API
    API_HOST=0.0.0.0
    API_PORT=8100
    DEBUG=True

    # Storage
    UPLOAD_DIR=./uploads
    MAX_FILE_SIZE=52428800
    ALLOWED_EXTENSIONS=[".pdf", ".docx"]

    # Qdrant
    QDRANT_HOST=localhost
    QDRANT_PORT=6333

    # API Externa
    COLLECTIONS_API_URL=http://localhost:9000/api

    # Embeddings
    EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
    EMBEDDING_DIMENSION=768

    # Text Processing
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=200
```
## 📁 Estructura del Proyecto

```bash
vectorizacion_final/
├── backend/
│   └── app/
│       ├── main.py              # API FastAPI principal
│       ├── services.py          # Servicios de procesamiento
│       ├── settings.py          # Configuración
│       └── __pycache__/
├── frontend/
│   └── interfaz.html            # Panel web
├── uploads/                     # Documentos temporales
├── .env                         # Variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 Uso
Iniciar Backend
```bash
cd backend/app
python main.py
```
La API estará disponible en: http://localhost:8100
Documentación interactiva: http://localhost:8100/docs
Acceder al Frontend
Abrir en navegador: interfaz.html (o servir con un servidor local)

```bash
# Con Python
python -m http.server 8000
# Luego acceder a: http://localhost:8000/frontend/interfaz.html
```

## 📡 Endpoints de la API
1. Validar y Subir Documentos

```bash
POST /api/upload-batch?collection_name=MiColeccion
Content-Type: multipart/form-data

Parámetros:
- files: Lista de archivos (PDF/DOCX)
- collection_name: Nombre de la colección destino

Respuesta:
{
  "batch_id": "uuid",
  "total_files": 5,
  "validated_files": 5,
  "failed_files": 0,
  "files": [
    {
      "file_id": "uuid",
      "filename": "documento.pdf",
      "total_pages": 10,
      "status": "validated",
      "is_duplicate": false
    }
  ]
}
```

2. Confirmar Reemplazo de Duplicados

```bash
POST /api/confirm-duplicates/{batch_id}?files_to_replace=file_id1&files_to_replace=file_id2

Respuesta:
{
  "success": true,
  "deleted_documents": [...],
  "message": "✅ 2 documento(s) antiguo(s) eliminado(s)"
}
```

3. Iniciar Vectorización

```bash
POST /api/vectorize-batch/{batch_id}?collection_name=MiColeccion

Respuesta:
{
  "message": "Vectorización de batch iniciada",
  "batch_id": "uuid",
  "collection_name": "MiColeccion",
  "total_files": 5
}
```

4. Consultar Progreso

```bash
GET /api/batch-progress/{batch_id}

Respuesta:
{
  "batch_id": "uuid",
  "total_files": 5,
  "completed_files": 3,
  "failed_files": 0,
  "overall_progress": 60,
  "current_file": "documento_3.pdf",
  "files_status": [
    {
      "filename": "documento.pdf",
      "status": "completed",
      "progress": 100
    }
  ]
}
```

5. Búsqueda Semántica

```bash
POST /api/search?collection_name=MiColeccion
Content-Type: application/json

Body:
{
  "query": "¿Qué es la metacognición?",
  "top_k": 5
}

Respuesta:
{
  "query": "¿Qué es la metacognición?",
  "collection": "MiColeccion",
  "results": [
    {
      "score": 0.85,
      "filename": "documento.pdf",
      "chunk": 5,
      "text": "La metacognición es..."
    }
  ]
}
```

6. Health Check

```bash
GET /health

Respuesta:
{
  "status": "healthy",
  "service": "document-vectorization-multiupload",
  "version": "2.0.0"
}
```

## 🎨 Flujo de la Interfaz
1. Selección de Colección: Elige una colección existente
2. Carga de Documentos: Arrastra o selecciona archivos PDF/DOCX
3. Validación: El sistema verifica estructura y detecta duplicados
4. Confirmación de Duplicados (si aplica): Decide si reemplazar documentos antiguos
5. Vectorización: Se procesan y vectorizan los documentos
6. Búsqueda: Realiza búsquedas semánticas en la colección

## 🔍 Nomenclatura de Archivos
Para un mejor seguimiento, se recomienda usar la siguiente estructura:

```bash
Nombre_del_programa_Curso#_Unidad#_Sesión#.pdf
```
Ejemplo:

```bash
Aprendo_en_Casa_Curso1_Unidad2_Sesión3.pdf
```

## ⚙️ Configuración Avanzada
Cambiar Modelo de Embeddings
En .env:
```bash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384
```
Modelos soportados:

* BAAI/bge-base-en-v1.5 (768 dimensiones) - Predeterminado
* sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensiones)
* sentence-transformers/all-mpnet-base-v2 (768 dimensiones)
Ajustar Tamaño de Chunks
En .env:
```bash
CHUNK_SIZE=1500      # Caracteres por fragmento
CHUNK_OVERLAP=300    # Solapamiento entre fragmentos
```

## 🐛 Solución de Problemas

Error: "La colección no existe en Qdrant"
* Asegúrate de crear la colección primero en la API de Colecciones
* Verifica que Qdrant esté corriendo: http://localhost:6333

Error: "Modelo no encontrado"
* El modelo se descargará automáticamente la primera vez
* Requiere conexión a internet y ~400 MB de espacio

Error: "CORS bloqueado"
* El backend ya tiene CORS habilitado para todas las orígenes
* Verifica que el frontend esté accediendo a http://localhost:8100

Archivos no se procesan
* Verifica que los archivos sean válidos PDF o DOCX
* Comprueba el tamaño máximo: 50 MB
* Revisa los logs del backend

## 📊 Monitoreo
Logs del Backend

```bash
# Ver logs en tiempo real
tail -f backend/app/main.py output

# Nivel de debug
DEBUG=True python main.py
```
Estado de Qdrant
```bash
# Verificar conexión
curl http://localhost:6333/health
```
## 🔐 Seguridad
- ✅ Validación de tipos de archivo
- ✅ Límite de tamaño de carga (50 MB)
- ✅ Sanitización de nombres de archivo
- ✅ CORS configurado
- ⚠️ DEBUG=False en producción

## 📈 Rendimiento

| Operación | Tiempo Estimado |
|--------------|--------------|
| Validación de documento | 2-5 segundos | 
| Vectorización (1000 chunks) | 30-60 segundos | 
| Búsqueda semántica | 100-500 ms |
| Carga de 10 documentos | 5-10 minutos |
