import json
import re
import os
import requests

# Import all AI provider libraries with fallbacks
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

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
        analysis_prompt = f"""# ROLE: Expert Prompt Engineer & AI Interaction Specialist

You are a world-class prompt engineering expert with deep knowledge of:
- Large Language Model behavior and optimization
- Chain-of-thought reasoning techniques  
- Few-shot learning and in-context learning
- Cognitive load theory in AI interactions
- Advanced prompting strategies (CoT, ReAct, Tree of Thoughts)

# TASK: Comprehensive Prompt Analysis & Enhancement

## INPUT PROMPT TO ANALYZE:
```
{query}
```

## ANALYSIS PROCESS:
Think step-by-step through each evaluation criterion, then provide your assessment.

### EVALUATION FRAMEWORK (Total: 100 points)

**1. CLARITY & PRECISION (25 points)**
- Language clarity and unambiguous instructions
- Specific, measurable, achievable, relevant, time-bound (SMART) criteria
- Absence of vague terms ("good", "better", "creative")
- Clear success metrics defined

**2. CONTEXT & BACKGROUND (20 points)**  
- Sufficient domain context provided
- Target audience clearly defined
- Use case and constraints specified
- Relevant examples or references included

**3. STRUCTURE & ORGANIZATION (20 points)**
- Logical flow and hierarchy
- Proper use of formatting (bullets, numbers, sections)
- Step-by-step instructions when needed
- Clear input/output format specifications

**4. ROLE & PERSONA DEFINITION (15 points)**
- Clear role assignment for the AI
- Appropriate expertise level specified
- Behavioral guidelines and tone definition
- Relevant background knowledge activation

**5. CONSTRAINTS & GUARDRAILS (10 points)**
- Output length and format requirements
- Ethical and safety boundaries
- Scope limitations clearly defined
- Edge case handling instructions

**6. ADVANCED TECHNIQUES (10 points)**
- Use of chain-of-thought reasoning prompts
- Few-shot examples when appropriate
- Self-reflection and validation requests
- Multi-step reasoning instructions

## ENHANCED PROMPT CREATION GUIDELINES:

### STRUCTURE TEMPLATE:
```
# ROLE: [Specific expert persona]
You are [detailed role with expertise areas]

# CONTEXT: [Situation/Background]
[Domain context, audience, use case]

# TASK: [Clear objective]
[Specific, measurable goal]

# INSTRUCTIONS: [Step-by-step process]
1. [First action with details]
2. [Second action with details]
...

# OUTPUT FORMAT: [Exact specifications]
[Structure, length, style requirements]

# CONSTRAINTS: [Limitations and boundaries]
- [Specific constraint 1]
- [Specific constraint 2]

# EXAMPLES: [If beneficial]
[Relevant few-shot examples]
```

### ENHANCEMENT STRATEGIES:
- **Specificity**: Replace vague terms with precise requirements
- **Context**: Add relevant background and domain knowledge
- **Structure**: Organize with clear hierarchy and formatting
- **Examples**: Include few-shot demonstrations when helpful
- **Constraints**: Define boundaries and limitations explicitly
- **Validation**: Add self-checking and reasoning requirements
- **Chain-of-Thought**: Encourage step-by-step reasoning
- **Output Format**: Specify exact format, length, and style

## REQUIRED OUTPUT FORMAT:
Respond with ONLY this exact JSON structure:

{{
    "score": [integer from 1-100],
    "clarity_score": [integer from 1-25], 
    "context_score": [integer from 1-20],
    "structure_score": [integer from 1-20],
    "role_score": [integer from 1-15],
    "constraints_score": [integer from 1-10],
    "advanced_score": [integer from 1-10],
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
    "improvements": ["improvement 1", "improvement 2", "improvement 3"],
    "new_prompt": "[complete enhanced prompt following best practices]",
    "reasoning": "Brief explanation of the score and main improvements made"
}}

## CONSTRAINTS:
- Provide ONLY the JSON response, no additional text
- Ensure all scores sum to the total score
- Enhanced prompt must be significantly improved, not just cosmetically changed
- Focus on practical, implementable improvements
- Consider the intended use case and context"""
        
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
                
                # Extract comprehensive analysis data
                result = {
                    "overall_score": parsed_response.get("score", 75),
                    "detailed_scores": {
                        "clarity": parsed_response.get("clarity_score", 18),
                        "context": parsed_response.get("context_score", 15),
                        "structure": parsed_response.get("structure_score", 15),
                        "role": parsed_response.get("role_score", 12),
                        "constraints": parsed_response.get("constraints_score", 8),
                        "advanced": parsed_response.get("advanced_score", 7)
                    },
                    "strengths": parsed_response.get("strengths", ["Basic functionality"]),
                    "weaknesses": parsed_response.get("weaknesses", ["Needs improvement"]),
                    "improvements": parsed_response.get("improvements", ["Add more detail"]),
                    "new_prompt": parsed_response.get("new_prompt", query),
                    "reasoning": parsed_response.get("reasoning", "Analysis completed")
                }
                
                return result["overall_score"], result
            else:
                # Fallback if JSON parsing fails
                fallback_result = {
                    "overall_score": 75,
                    "detailed_scores": {"clarity": 18, "context": 15, "structure": 15, "role": 12, "constraints": 8, "advanced": 7},
                    "strengths": ["Prompt provided"],
                    "weaknesses": ["Analysis failed"],
                    "improvements": ["Retry analysis"],
                    "new_prompt": f"Enhanced version: {query}",
                    "reasoning": "JSON parsing failed, using fallback"
                }
                return 75, fallback_result
                
        except json.JSONDecodeError:
            # If parsing fails, try to extract score and prompt manually
            score_match = re.search(r'"score":\s*(\d+)', response_content)
            prompt_match = re.search(r'"new_prompt":\s*"([^"]*)"', response_content)
            
            score = int(score_match.group(1)) if score_match else 75
            new_prompt = prompt_match.group(1) if prompt_match else f"Enhanced version: {query}"
            
            fallback_result = {
                "overall_score": score,
                "detailed_scores": {"clarity": 18, "context": 15, "structure": 15, "role": 12, "constraints": 8, "advanced": 7},
                "strengths": ["Basic prompt structure"],
                "weaknesses": ["Analysis parsing issues"],
                "improvements": ["Improve prompt format"],
                "new_prompt": new_prompt,
                "reasoning": "Partial parsing successful"
            }
            
            return score, fallback_result
        
    except Exception as e:
        # Return error information instead of crashing
        error_msg = f"Error in prompt analysis: {str(e)}"
        return "Error", error_msg


