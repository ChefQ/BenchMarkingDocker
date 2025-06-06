from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import uvicorn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers.pipelines import pipeline
import torch
import logging
import time
from contextlib import asynccontextmanager
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
model_cache = {}
tokenizer_cache = {}

class InferenceRequest(BaseModel):
    model_name: str = Field(..., description="HuggingFace model name (e.g., 'bert-base-uncased')")
    text: str = Field(..., description="Input text for inference")
    task: Optional[str] = Field(default="feature-extraction", description="Task type: feature-extraction, text-classification, text-generation, etc.")
    max_length: Optional[int] = Field(default=512, description="Maximum sequence length")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional model parameters")

class InferenceResponse(BaseModel):
    model_name: str
    result: Any
    processing_time: float
    task: str

class ModelInfo(BaseModel):
    loaded_models: List[str]
    cache_size: int
    device: str

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

async def load_model_and_tokenizer(model_name: str, task: str):
    """Load model and tokenizer with caching"""
    cache_key = f"{model_name}_{task}"
    
    if cache_key in model_cache:
        logger.info(f"Using cached model: {model_name}")
        return model_cache[cache_key], tokenizer_cache.get(model_name)
    
    try:
        logger.info(f"Loading model: {model_name} for task: {task}")
        device = get_device()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/app/cache")
        tokenizer_cache[model_name] = tokenizer
        
        # Load model based on task
        if task == "text-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="/app/cache")
        elif task == "text-generation":
            model = AutoModelForCausalLM.from_pretrained(model_name , cache_dir="/app/cache")
        else:
            # Default to AutoModel for feature extraction
            model = AutoModel.from_pretrained(model_name, cache_dir="/app/cache")
        
        model = model.to(device)
        model.eval()
        
        model_cache[cache_key] = model
        logger.info(f"Successfully loaded model: {model_name} on {device}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")

async def perform_inference(model, tokenizer, text: str, task: str, max_length: int, parameters: Dict):
    """Perform inference based on task type"""
    device = get_device()
    
    try:
        if task == "text-generation":
            # Use pipeline for text generation
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                **parameters
            )
            result = pipe(text, max_length=max_length, num_return_sequences=1)
            if isinstance(result, list) and len(result) > 0:
                return result[0]["generated_text"]
            else:
                raise HTTPException(status_code=500, detail="Text generation failed to return valid output.")
            
        elif task == "text-classification":
            # Use pipeline for classification
            pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                **parameters
            )
            result = pipe(text)
            return result
            
        else:
            # Feature extraction (default)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Return last hidden state mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy().tolist()
            
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting HuggingFace Inference Server")
    logger.info(f"Using device: {get_device()}")
    yield
    # Shutdown
    logger.info("Shutting down server")
    # Clear model cache
    global model_cache, tokenizer_cache
    model_cache.clear()
    tokenizer_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="HuggingFace Model Inference Server",
    description="A scalable inference server for any HuggingFace model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "HuggingFace Model Inference Server",
        "status": "running",
        "device": get_device()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/models", response_model=ModelInfo)
async def get_loaded_models():
    return ModelInfo(
        loaded_models=list(model_cache.keys()),
        cache_size=len(model_cache),
        device=get_device()
    )

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Main inference endpoint"""
    start_time = time.time()

    # if request arguments are not valid, raise an error
    if request.task not in ["feature-extraction", "text-classification", "text-generation"]:
        raise HTTPException(status_code=400, detail="Invalid task type. Supported tasks: feature-extraction, text-classification, text-generation.")
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text is required.")
    if request.max_length is not None and (not isinstance(request.max_length, int) or request.max_length <= 0): 
        raise HTTPException(status_code=400, detail="max_length must be a positive integer.")
    if request.parameters is not None and not isinstance(request.parameters, dict):
        raise HTTPException(status_code=400, detail="parameters must be a dictionary.")

    try:
        # Load model and tokenizer
        model, tokenizer = await load_model_and_tokenizer(request.model_name, request.task)
        
        # Perform inference
        result = await perform_inference(
            model=model,
            tokenizer=tokenizer,
            text=request.text,
            task= request.task,
            max_length=request.max_length if request.max_length is not None else 512,
            parameters=request.parameters if request.parameters is not None else {}
        )
        
        processing_time = time.time() - start_time
        
        return InferenceResponse(
            model_name=request.model_name,
            result=result,
            processing_time=processing_time,
            task=request.task
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[InferenceRequest]):
    """Batch inference endpoint"""
    start_time = time.time()
    
    try:
        # Process requests concurrently
        tasks = [predict(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "request_index": i
                })
            else:
                processed_results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            "results": processed_results,
            "total_processing_time": total_time,
            "batch_size": len(requests)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model from cache"""
    removed_keys = []
    keys_to_remove = [key for key in model_cache.keys() if model_name in key]
    
    for key in keys_to_remove:
        del model_cache[key]
        removed_keys.append(key)
    
    if model_name in tokenizer_cache:
        del tokenizer_cache[model_name]
        removed_keys.append(f"tokenizer_{model_name}")
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "message": f"Unloaded model components: {removed_keys}",
        "remaining_models": list(model_cache.keys())
    }

if __name__ == "__main__":

    # unicorn is needed to run the FastAPI app
    # FastAPI is just a framework, it needs a server to run
    # uvicorn is a lightning-fast ASGI server, perfect for FastAPI
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker to share model cache
        reload=False
    )