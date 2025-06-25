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
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None
    Anthropic = None

# Provider configurations
PROVIDER_CONFIGS = {
    "groq": {
        "name": "Groq",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile", 
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b"
        ],
        "default_model": "llama-3.3-70b-versatile"
    },
    "openai": {
        "name": "OpenAI",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ],
        "default_model": "gpt-4o-mini"
    },
    "anthropic": {
        "name": "Anthropic",
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        "default_model": "claude-3-5-haiku-20241022"
    },
    "mistral": {
        "name": "Mistral AI",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b"
        ],
        "default_model": "mistral-small-latest"
    },
    "together": {
        "name": "Together AI",
        "models": [
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "Qwen/Qwen2.5-72B-Instruct-Turbo"
        ],
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    },
    "openrouter": {
        "name": "OpenRouter",
        "models": [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.1-70b-instruct",
            "mistralai/mistral-large",
            "google/gemini-pro-1.5"
        ],
        "default_model": "openai/gpt-4o-mini"
    }
}

def get_analysis_prompt(query, style="comprehensive"):
    """Get the analysis prompt based on the selected style"""
    
    if style == "quick":
        return _get_quick_analysis_prompt(query)
    elif style == "detailed":
        return _get_detailed_analysis_prompt(query)
    else:  # comprehensive (default)
        return _get_comprehensive_analysis_prompt(query)

def _get_comprehensive_analysis_prompt(query):
    """Comprehensive analysis - enhanced with advanced prompt engineering principles"""
    return f"""# ASSIGNED ROLE: Expert Prompt Engineer & AI System Designer

You are a world-class prompt engineering specialist with expertise in:
- Advanced prompt architecture and optimization
- Chain-of-Thought (CoT) and Few-shot prompting techniques
- System prompt design and AI persona development
- Cognitive load theory and human-AI interaction patterns
- Multi-step reasoning and self-consistency methods
- Iterative refinement and adaptive prompt structures

# TASK OBJECTIVE: 
Analyze the given prompt using proven prompt engineering patterns and generate a comprehensive, professionally enhanced version that maximizes clarity, effectiveness, and output quality.

# INPUT PROMPT TO ANALYZE:
```
{query}
```

# STRUCTURED ANALYSIS PROCESS:

## STEP 1: EVALUATION FRAMEWORK (100 points total)
Apply advanced Chain-of-Thought reasoning for each criterion:

**CLARITY & PRECISION (25 points)**
- Unambiguous instructions with clear delimiters
- SMART criteria implementation (Specific, Measurable, Achievable, Relevant, Time-bound)
- Elimination of vague qualifiers and subjective terms
- Explicit success metrics and validation criteria

**CONTEXT & BACKGROUND (20 points)**
- Domain expertise and background knowledge activation
- Target audience and use case specification
- Relevant constraints and environmental factors
- Reference examples and contextual anchors

**STRUCTURE & ORGANIZATION (20 points)**
- Hierarchical information architecture
- Strategic use of formatting (sections, bullets, numbering)
- Logical flow with clear transitions
- Input/output format specifications with examples

**ROLE & PERSONA DEFINITION (15 points)**
- Specific expert persona assignment
- Behavioral guidelines and communication tone
- Knowledge domain activation
- Appropriate expertise level calibration

**CONSTRAINTS & GUARDRAILS (10 points)**
- Output specifications (length, format, style)
- Scope boundaries and limitation definitions
- Quality control and validation requirements
- Edge case handling instructions

**ADVANCED TECHNIQUES (10 points)**
- Chain-of-Thought reasoning integration
- Few-shot learning examples when beneficial
- Self-reflection and meta-cognitive prompts
- Multi-step reasoning and iterative refinement

## STEP 2: ENHANCED PROMPT GENERATION

### STRUCTURAL TEMPLATE IMPLEMENTATION:
Apply this proven pattern for maximum effectiveness:

```
# ASSIGNED ROLE: [Specific expert persona with clear expertise areas]

# TASK OBJECTIVE: [Clear, measurable goal with success criteria]

# CONTEXT: [Domain background, audience, use case, constraints]

# STRUCTURED SECTIONS:
1. [Primary task breakdown]
2. [Secondary considerations]
3. [Quality assurance steps]

# FORMATTING GUIDELINES:
- [Specific output structure requirements]
- [Style and tone specifications]
- [Length and detail parameters]

# TONE AND STYLE: [Communication approach and persona consistency]

# FLEXIBILITY NOTES: [Adaptation and reuse considerations]
```

### BEST PRACTICES INTEGRATION:
- **Delimiters**: Use triple quotes, sections, and clear separators
- **Examples**: Include few-shot demonstrations for complex formats
- **Iterative Design**: Enable prompt refinement and adaptation
- **Pattern Recognition**: Incorporate proven prompt engineering templates
- **Advanced Methods**: Apply CoT, self-consistency, and multi-step reasoning

# OUTPUT FORMAT:
You MUST respond with ONLY this exact JSON structure. No additional text:

{{
    "score": [sum of all individual scores below],
    "clarity_score": [integer from 1-25], 
    "context_score": [integer from 1-20],
    "structure_score": [integer from 1-20],
    "role_score": [integer from 1-15],
    "constraints_score": [integer from 1-10],
    "advanced_score": [integer from 1-10],
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
    "improvements": ["improvement 1", "improvement 2", "improvement 3"],
    "new_prompt": "[MUST be completely rewritten using advanced prompt engineering principles - NOT the original]",
    "reasoning": "Brief explanation of scoring rationale and key enhancements applied"
}}

# CRITICAL REQUIREMENTS:
- Apply Chain-of-Thought reasoning throughout analysis
- Verify mathematical accuracy: total = sum of individual scores
- Enhanced prompt must demonstrate significant structural improvement
- Incorporate multiple prompt engineering best practices
- Transform basic requests into professional, comprehensive system prompts
- Ensure adaptability and reuse potential in the enhanced version

# ENHANCEMENT BENCHMARK:
Transform simple prompts like "write about X" into comprehensive system prompts with:
- Clear role assignment and expertise activation
- Structured task breakdown with measurable objectives  
- Specific formatting and style guidelines
- Context-appropriate constraints and quality measures
- Integration of advanced prompting techniques where beneficial

The enhanced prompt should be 5-10x more detailed and professionally structured than the original."""

def prompt_analysis(query, api_key, temp, max_token, provider="groq", model=None, style="comprehensive"):
    """Main prompt analysis function supporting multiple AI providers"""
    try:
        # Validate inputs
        if not query or not isinstance(query, str) or not query.strip():
            return "Error", "Please provide a valid prompt to analyze."
        
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            return "Error", "Please provide a valid API key."
        
        # Normalize provider name
        provider = provider.lower().strip()
        
        # Validate provider
        if provider not in PROVIDER_CONFIGS:
            return "Error", f"Unsupported provider: {provider}. Supported providers: {', '.join(PROVIDER_CONFIGS.keys())}"
        
        # Set default model if not provided
        if not model:
            model = PROVIDER_CONFIGS[provider]["default_model"]
        
        # Validate parameters
        try:
            temp = float(temp)
            if temp < 0 or temp > 2:
                temp = 0.7
        except (ValueError, TypeError):
            temp = 0.7
        
        try:
            max_token = int(max_token)
            if max_token < 1 or max_token > 32768:
                max_token = 1000
        except (ValueError, TypeError):
            max_token = 1000
        
        # Route to appropriate provider
        if provider == "groq":
            return _analyze_with_groq(query, api_key, temp, max_token, model, style)
        elif provider == "openai":
            return _analyze_with_openai(query, api_key, temp, max_token, model, style)
        elif provider == "anthropic":
            return _analyze_with_anthropic(query, api_key, temp, max_token, model, style)
        elif provider == "mistral":
            return _analyze_with_mistral(query, api_key, temp, max_token, model, style)
        elif provider == "together":
            return _analyze_with_together(query, api_key, temp, max_token, model, style)
        elif provider == "openrouter":
            return _analyze_with_openrouter(query, api_key, temp, max_token, model, style)
        else:
            return "Error", f"Provider implementation not found: {provider}"
    
    except Exception as e:
        return "Error", f"Error in prompt analysis: {str(e)}"

def _analyze_with_groq(query, api_key, temp, max_token, model, style):
    """Analyze prompt using Groq"""
    if not Groq:
        return "Error", "Groq library not installed. Run: pip install groq"
    
    try:
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": get_analysis_prompt(query, style)}
            ],
            temperature=temp,
            max_tokens=max_token,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return _parse_response(completion.choices[0].message.content, query)
    
    except Exception as e:
        return "Error", f"Groq API error: {str(e)}"

def _analyze_with_openai(query, api_key, temp, max_token, model, style):
    """Analyze prompt using OpenAI"""
    if not OpenAI:
        return "Error", "OpenAI library not installed. Run: pip install openai"
    
    try:
        client = OpenAI(api_key=api_key)
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": get_analysis_prompt(query, style)}
            ],
            temperature=temp,
            max_tokens=max_token,
            top_p=1,
        )
        
        return _parse_response(completion.choices[0].message.content, query)
    
    except Exception as e:
        return "Error", f"OpenAI API error: {str(e)}"

def _analyze_with_anthropic(query, api_key, temp, max_token, model, style):
    """Analyze prompt using Anthropic Claude"""
    if not Anthropic:
        return "Error", "Anthropic library not installed. Run: pip install anthropic"
    
    try:
        client = Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=max_token,
            temperature=temp,
            messages=[
                {"role": "user", "content": get_analysis_prompt(query, style)}
            ]
        )
        
        return _parse_response(message.content[0].text, query)
    
    except Exception as e:
        return "Error", f"Anthropic API error: {str(e)}"

def _analyze_with_mistral(query, api_key, temp, max_token, model, style):
    """Analyze prompt using Mistral AI"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": get_analysis_prompt(query, style)}
            ],
            "temperature": temp,
            "max_tokens": max_token
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            return "Error", f"Mistral API error: {response.status_code} - {response.text}"
        
        result = response.json()
        return _parse_response(result["choices"][0]["message"]["content"], query)
    
    except Exception as e:
        return "Error", f"Mistral API error: {str(e)}"

def _analyze_with_together(query, api_key, temp, max_token, model, style):
    """Analyze prompt using Together AI"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": get_analysis_prompt(query, style)}
            ],
            "temperature": temp,
            "max_tokens": max_token
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            return "Error", f"Together AI API error: {response.status_code} - {response.text}"
        
        result = response.json()
        return _parse_response(result["choices"][0]["message"]["content"], query)
    
    except Exception as e:
        return "Error", f"Together AI API error: {str(e)}"

def _analyze_with_openrouter(query, api_key, temp, max_token, model, style):
    """Analyze prompt using OpenRouter"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/prompt-analysis-tool",
            "X-Title": "Prompt Analysis Tool"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": get_analysis_prompt(query, style)}
            ],
            "temperature": temp,
            "max_tokens": max_token
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            return "Error", f"OpenRouter API error: {response.status_code} - {response.text}"
        
        result = response.json()
        return _parse_response(result["choices"][0]["message"]["content"], query)
    
    except Exception as e:
        return "Error", f"OpenRouter API error: {str(e)}"

def _parse_response(response_content, original_query):
    """Parse the AI response and extract analysis data"""
    try:
        # Clean up the response to extract JSON
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_response = json.loads(json_str)
            
            # Extract comprehensive analysis data with validation
            clarity = parsed_response.get("clarity_score", 5)
            context = parsed_response.get("context_score", 5)
            structure = parsed_response.get("structure_score", 5)
            role = parsed_response.get("role_score", 5)
            constraints = parsed_response.get("constraints_score", 3)
            advanced = parsed_response.get("advanced_score", 2)
            
            # Validate math and fix if needed
            calculated_total = clarity + context + structure + role + constraints + advanced
            reported_total = parsed_response.get("score", calculated_total)
            
            # Use calculated total if there's a math error
            if abs(calculated_total - reported_total) > 2:
                final_score = calculated_total
            else:
                final_score = reported_total
            
            result = {
                "overall_score": final_score,
                "detailed_scores": {
                    "clarity": clarity,
                    "context": context,
                    "structure": structure,
                    "role": role,
                    "constraints": constraints,
                    "advanced": advanced
                },
                "strengths": parsed_response.get("strengths", ["Basic functionality"]),
                "weaknesses": parsed_response.get("weaknesses", ["Needs improvement"]),
                "improvements": parsed_response.get("improvements", ["Add more detail"]),
                "new_prompt": _validate_enhanced_prompt(parsed_response.get("new_prompt", ""), original_query),
                "reasoning": parsed_response.get("reasoning", "Analysis completed"),
                "math_corrected": abs(calculated_total - reported_total) > 2
            }
            
            return result["overall_score"], result
        else:
            # Fallback if JSON parsing fails
            fallback_result = {
                "overall_score": 25,
                "detailed_scores": {"clarity": 8, "context": 4, "structure": 5, "role": 3, "constraints": 3, "advanced": 2},
                "strengths": ["Basic request provided"],
                "weaknesses": ["Analysis parsing failed", "Very basic prompt structure"],
                "improvements": ["Add specific context", "Define target audience", "Specify requirements"],
                "new_prompt": _create_basic_enhancement(original_query),
                "reasoning": "JSON parsing failed, using fallback analysis"
            }
            return 25, fallback_result
            
    except json.JSONDecodeError:
        # If parsing fails, try to extract score and prompt manually
        score_match = re.search(r'"score":\s*(\d+)', response_content)
        prompt_match = re.search(r'"new_prompt":\s*"([^"]*)"', response_content)
        
        score = int(score_match.group(1)) if score_match else 75
        new_prompt = prompt_match.group(1) if prompt_match else f"Enhanced version: {original_query}"
        
        fallback_result = {
            "overall_score": score if score < 50 else 25,
            "detailed_scores": {"clarity": 6, "context": 3, "structure": 4, "role": 3, "constraints": 2, "advanced": 2},
            "strengths": ["Basic request provided"],
            "weaknesses": ["Analysis parsing issues", "Very basic prompt"],
            "improvements": ["Add specific context", "Define requirements", "Specify format"],
            "new_prompt": _validate_enhanced_prompt(new_prompt, original_query),
            "reasoning": "Partial parsing successful, scores adjusted for basic prompt"
        }
        
        return score, fallback_result

def get_provider_models(provider):
    """Get available models for a provider"""
    return PROVIDER_CONFIGS.get(provider, {}).get("models", [])

def get_supported_providers():
    """Get list of supported providers"""
    return list(PROVIDER_CONFIGS.keys())

def _get_quick_analysis_prompt(query):
    """Quick analysis - fast and focused"""
    return f"""# ROLE: Senior Prompt Engineer

You are a senior prompt engineer focused on rapid assessment and immediate improvements.

# TASK: Quick Prompt Analysis

## INPUT PROMPT:
```
{query}
```

## QUICK EVALUATION:
Analyze this prompt quickly and provide immediate feedback focusing on the most critical issues.

### SCORING CRITERIA (Total: 100 points):
1. **Clarity** (30 points) - Is it clear what's being asked?
2. **Specificity** (25 points) - Are requirements specific enough?
3. **Context** (20 points) - Is there sufficient background?
4. **Structure** (15 points) - Is it well organized?
5. **Completeness** (10 points) - Are key elements present?

## OUTPUT FORMAT:
Return ONLY this JSON (no other text):

{{
    "score": [sum of all scores below],
    "clarity_score": [1-25],
    "context_score": [1-20], 
    "structure_score": [1-20],
    "role_score": [1-15],
    "constraints_score": [1-10],
    "advanced_score": [1-10],
    "strengths": ["key strength 1", "key strength 2"],
    "weaknesses": ["main weakness 1", "main weakness 2"],
    "improvements": ["quick fix 1", "quick fix 2", "quick fix 3"],
    "new_prompt": "[completely rewritten and enhanced version - must be significantly different from original]",
    "reasoning": "Brief 1-sentence explanation of the main issue and improvement"
}}

SCORING RULES:
- Total = clarity_score + context_score + structure_score + role_score + constraints_score + advanced_score
- Basic prompts typically score 15-35 total
- Focus on the most impactful improvements that can be made quickly."""

def _get_detailed_analysis_prompt(query):
    """Detailed analysis - thorough and comprehensive"""
    return f"""# ROLE: Master Prompt Engineering Consultant & AI Optimization Expert

You are a world-renowned prompt engineering consultant with 10+ years of experience optimizing AI interactions across industries. Your expertise includes:
- Advanced LLM behavioral psychology and cognitive architectures
- Prompt engineering methodologies (CoT, ToT, ReAct, Constitutional AI)
- Cross-cultural AI communication patterns
- Enterprise-grade prompt optimization strategies
- Multi-modal and complex reasoning prompt design
- AI safety and alignment in prompt engineering

# TASK: Deep Comprehensive Prompt Analysis & Strategic Enhancement

## INPUT PROMPT FOR DETAILED ANALYSIS:
```
{query}
```

## COMPREHENSIVE EVALUATION FRAMEWORK:

### PRIMARY ASSESSMENT DIMENSIONS (100 points total):

**1. CLARITY & LINGUISTIC PRECISION (25 points)**
- Semantic clarity and disambiguation
- Terminology consistency and domain appropriateness  
- Instruction hierarchy and logical flow
- Ambiguity identification and resolution
- Cross-cultural communication considerations

**2. CONTEXTUAL ARCHITECTURE (20 points)**
- Domain expertise and background activation
- Stakeholder and audience alignment
- Environmental and situational context
- Historical and temporal context integration
- Cross-reference and knowledge base utilization

**3. STRUCTURAL ENGINEERING (20 points)**
- Information architecture and flow design
- Modular prompt component organization
- Hierarchical instruction design
- Template and pattern utilization
- Scalability and maintainability factors

**4. PERSONA & ROLE OPTIMIZATION (15 points)**
- AI agent persona definition and consistency
- Expertise level calibration and specialization
- Behavioral guideline specification
- Communication style and tone alignment
- Professional and ethical framework integration

**5. CONSTRAINT ARCHITECTURE (10 points)**
- Output specification and formatting requirements
- Boundary conditions and limitation definitions
- Error handling and edge case management
- Quality assurance and validation criteria
- Compliance and safety constraint integration

**6. ADVANCED METHODOLOGY INTEGRATION (10 points)**
- Chain-of-thought reasoning implementation
- Few-shot and in-context learning optimization
- Meta-cognitive prompting techniques
- Self-reflection and validation mechanisms
- Multi-step reasoning and decomposition strategies

## STRATEGIC ENHANCEMENT FRAMEWORK:

### OPTIMIZATION STRATEGIES:
- **Cognitive Load Management**: Reduce complexity while maintaining depth
- **Attention Direction**: Guide AI focus to critical elements
- **Context Priming**: Activate relevant knowledge domains
- **Output Shaping**: Structure responses for maximum utility
- **Error Prevention**: Anticipate and prevent common failure modes
- **Performance Scaling**: Optimize for different AI model capabilities

### INDUSTRY BEST PRACTICES INTEGRATION:
- Academic research prompt optimization
- Enterprise deployment considerations
- Multi-stakeholder prompt design
- Cross-platform compatibility
- Version control and iterative improvement

## REQUIRED DETAILED OUTPUT:
Provide comprehensive analysis in this exact JSON format:

{{
    "score": [integer 1-100],
    "clarity_score": [integer 1-25],
    "context_score": [integer 1-20],
    "structure_score": [integer 1-20], 
    "role_score": [integer 1-15],
    "constraints_score": [integer 1-10],
    "advanced_score": [integer 1-10],
    "strengths": ["detailed strength 1", "detailed strength 2", "detailed strength 3", "detailed strength 4"],
    "weaknesses": ["detailed weakness 1", "detailed weakness 2", "detailed weakness 3", "detailed weakness 4"],
    "improvements": ["strategic improvement 1", "strategic improvement 2", "strategic improvement 3", "strategic improvement 4", "strategic improvement 5"],
    "new_prompt": "[completely optimized prompt with advanced techniques, structured components, and professional-grade enhancements]",
    "reasoning": "Detailed analysis of the current prompt's architecture, identification of strategic optimization opportunities, and explanation of the comprehensive enhancement approach implemented",
    "methodology_notes": "Advanced techniques used in the enhancement (CoT, few-shot, etc.)",
    "use_case_analysis": "Assessment of the prompt's intended use case and optimization for that specific context",
    "scalability_assessment": "Evaluation of how this prompt will perform across different AI models and use cases"
}}

## CONSTRAINTS:
- Provide thorough, expert-level analysis
- Enhanced prompt must be significantly more sophisticated
- Include cutting-edge prompt engineering techniques
- Consider enterprise-grade implementation requirements
- Focus on maximum effectiveness and professional deployment readiness"""

def _validate_enhanced_prompt(enhanced_prompt, original_query):
    """Validate that the enhanced prompt is actually enhanced, not just the original"""
    if not enhanced_prompt or enhanced_prompt.strip() == "":
        # Create a basic enhanced version if none provided
        return _create_basic_enhancement(original_query)
    
    # Check if enhanced prompt is too similar to original (similarity check)
    original_clean = original_query.lower().strip()
    enhanced_clean = enhanced_prompt.lower().strip()
    
    # If enhanced prompt is identical or too similar, create a better one
    if enhanced_clean == original_clean or len(enhanced_prompt) < len(original_query) * 2:
        return _create_basic_enhancement(original_query)
    
    return enhanced_prompt

def _create_basic_enhancement(original_query):
    """Create a basic enhanced version when AI fails to provide one"""
    original_clean = original_query.strip().lower()
    
    # Handle common patterns
    if "write" in original_clean and "blog" in original_clean:
        if "virat" in original_clean:
            return """You are a professional sports journalist. Write a comprehensive 1000-word blog post about Virat Kohli for cricket enthusiasts. 

Structure your post with:
1. Introduction to Virat's significance in cricket
2. Career highlights and achievements
3. Playing style and techniques
4. Leadership journey as captain
5. Impact on Indian cricket
6. Recent performances and statistics
7. Conclusion about his legacy

Requirements:
- Use engaging, informative tone
- Include specific statistics and examples
- Target audience: cricket fans aged 20-50
- Add subheadings for better readability
- End with a thought-provoking conclusion"""
    
    # Generic enhancement for other prompts
    elif "write" in original_clean:
        return f"""You are a professional content writer. Create a well-structured, comprehensive piece based on: "{original_query}"

Requirements:
- Define your target audience
- Include specific examples and details
- Use clear structure with headings
- Specify word count (800-1200 words)
- Choose appropriate tone and style
- Add relevant context and background
- Include actionable insights or conclusions"""
    
    # Fallback enhancement
    return f"""Enhanced version of: "{original_query}"

You are an expert in the relevant field. Please provide a detailed, well-structured response that includes:
- Clear context and background
- Specific examples and evidence
- Target audience consideration
- Appropriate length and format
- Professional tone and style
- Actionable insights or recommendations

Structure your response with clear sections and provide comprehensive coverage of the topic."""