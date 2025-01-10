import time
import backoff

def chunk_text(text, max_tokens=1024):
    """Split text into chunks that respect token limits"""
    # Simple splitting by sentences or paragraphs
    # You can adjust the chunk size as needed
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_tokens:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + "."
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

@backoff.on_exception(backoff.expo, 
                     (TimeoutError, Exception),
                     max_tries=3,
                     max_time=30)
def process_with_rate_limit(text):
    chunks = chunk_text(text, max_tokens=1024)
    responses = []
    
    for chunk in chunks:
        try:
            # Process chunk with Groq API with timeout handling
            response = completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": chunk
                    }
                ],
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=1024,
                timeout=30,  # Add timeout parameter
            )
            responses.append(response)
            
            # Increased delay between requests
            time.sleep(2)  # Increased from 1 to 2 seconds
            
        except Exception as e:
            if "timeout" in str(e).lower():
                print(f"Timeout error: {e}. Retrying with backoff...")
                raise TimeoutError(str(e))
            else:
                print(f"Error processing chunk: {e}")
            
    return responses 