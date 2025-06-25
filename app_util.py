import json
import re
from groq import Groq

def prompt_analysis(query, api_key, temp, max_token):
    try:
        # Validate inputs
        if not api_key or not isinstance(api_key, str):
            return "Error", "Valid API key is required"
        
        if not query or not isinstance(query, str):
            return "Error", "Valid query is required"
        
        # Ensure temperature is a valid float
        try:
            temp = float(temp)
            if temp < 0 or temp > 2:
                temp = 0.7
        except (ValueError, TypeError):
            temp = 0.7
        
        # Ensure max_tokens is a valid integer
        try:
            max_token = int(max_token)
            if max_token < 1 or max_token > 32768:
                max_token = 1000
        except (ValueError, TypeError):
            max_token = 1000
        
        # Initialize the Groq client
        client = Groq(api_key=api_key)
        
        # Create the analysis prompt
        analysis_prompt = f"""You are an expert prompt engineer. Analyze the following prompt and provide a detailed evaluation:

PROMPT TO ANALYZE: "{query}"

Please provide your analysis in the following JSON format:
{{
    "score": [number from 1-100],
    "new_prompt": "[improved version of the prompt]"
}}

Scoring criteria (1-100):
- Clarity and specificity (25 points)
- Context and background information (20 points)
- Clear instructions and objectives (20 points)
- Appropriate complexity and scope (15 points)
- Use of explicit cues and constraints (10 points)
- Overall effectiveness (10 points)

For the improved prompt:
- Make it clearer and more specific
- Add relevant context if missing
- Structure instructions better
- Add constraints to limit scope if needed
- Use explicit cues to guide the response

Respond ONLY with the JSON format, no other text."""
        
        # Make the API call with proper parameter names
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=temp,
            max_tokens=max_token
        )
        
        # Get the response content
        response_content = completion.choices[0].message.content
        
        # Try to parse the JSON response
        try:
            # Clean up the response to extract JSON
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_response = json.loads(json_str)
                
                score = parsed_response.get("score", 75)
                new_prompt = parsed_response.get("new_prompt", query)
                
                return score, new_prompt
            else:
                # Fallback if JSON parsing fails
                return 75, f"Improved version: {query}"
                
        except json.JSONDecodeError:
            # If parsing fails, try to extract score and prompt manually
            score_match = re.search(r'"score":\s*(\d+)', response_content)
            prompt_match = re.search(r'"new_prompt":\s*"([^"]*)"', response_content)
            
            score = int(score_match.group(1)) if score_match else 75
            new_prompt = prompt_match.group(1) if prompt_match else f"Improved version: {query}"
            
            return score, new_prompt
        
    except Exception as e:
        # Return error information instead of crashing
        error_msg = f"Error in prompt analysis: {str(e)}"
        return "Error", error_msg


