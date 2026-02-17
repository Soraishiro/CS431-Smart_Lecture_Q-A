"""Gemini client for answer generation using Vertex AI."""

import logging
import time
from typing import List, Dict, Any
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for generating answers using Gemini 2.5 via Vertex AI."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize the Gemini client.
        
        Args:
            project_id: GCP project ID
            location: GCP location for Vertex AI (default: us-central1)
            model: Gemini model name (default: gemini-2.0-flash-exp)
            
        Raises:
            Exception: If initialization fails
        """
        try:
            logger.info(f"Initializing Gemini client with project={project_id}, location={location}, model={model}")
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=location)
            
            # Initialize the generative model
            self.model = GenerativeModel(model)
            self.project_id = project_id
            self.location = location
            self.model_name = model
            
            logger.info(f"Successfully initialized Gemini client with model: {model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise Exception(f"Failed to initialize GeminiClient: {e}") from e
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 512,
        max_retries: int = 3
    ) -> str:
        """
        Generate an answer using Gemini based on query and contexts.
        
        Args:
            query: User query text
            contexts: List of top-k chunks (typically top 5) with text, timestamps, etc.
            temperature: Sampling temperature (default: 0.3)
            max_tokens: Maximum output tokens (default: 512)
            max_retries: Maximum retry attempts (default: 3)
            
        Returns:
            Generated answer without timestamp citations
            
        Raises:
            Exception: If generation fails after all retries
        """
        # Construct the prompt
        prompt = self._construct_prompt(query, contexts)
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating answer (attempt {attempt + 1}/{max_retries})")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract text from response
                answer = response.text
                
                logger.info(f"Successfully generated answer ({len(answer)} chars)")
                return answer
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} generation attempts failed")
                    raise Exception(f"Failed to generate answer after {max_retries} attempts: {e}") from e
    
    def _construct_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Construct the prompt for Gemini with query and contexts.
        
        Args:
            query: User query text
            contexts: List of context chunks with text, enhanced, timestamps
            
        Returns:
            Formatted prompt string
        """
        # Take top 5 contexts
        top_contexts = contexts[:5]
        
        # Format contexts (keep timestamps for model context, but won't ask to cite them)
        context_text = ""
        for i, ctx in enumerate(top_contexts, 1):
            # Use enhanced text for LLM (better comprehension)
            # Fall back to original text if enhanced not available
            text = ctx.get("enhanced", "") or ctx.get("text", "")
            
            context_text += f"\n[Ngữ cảnh {i}]\n"
            context_text += f"{text}\n"
        
        # Construct the full prompt - straightforward Vietnamese instructions
        prompt = f"""Bạn là trợ lý AI trả lời câu hỏi về Machine Learning và Deep Learning.

QUY TẮC QUAN TRỌNG:
- Trả lời NGẮN GỌN và TRỰC TIẾP vào vấn đề
- CHỈ sử dụng thông tin từ các ngữ cảnh bên dưới
- KHÔNG thêm kiến thức bên ngoài
- KHÔNG cần nói "Dựa vào bài giảng" hay "Theo ngữ cảnh"
- KHÔNG cần trích dẫn thời gian
- Nếu ngữ cảnh KHÔNG đủ thông tin, nói rõ "Thông tin không đề cập trong bài giảng"

Câu hỏi: {query}

Ngữ cảnh:
{context_text}

Trả lời:"""
        
        return prompt


    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.3
    ) -> str:
        """
        Alias for generate() method to match KaggleGeminiClient interface.
        
        Args:
            query: User query
            context_chunks: List of context dictionaries
            temperature: Generation temperature
            
        Returns:
            Generated answer text
        """
        return self.generate(query, context_chunks, temperature=temperature)
