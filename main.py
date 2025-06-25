from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import os
import json
from app_util import prompt_analysis

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Enable CORS for all routes
CORS(app)

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Prompt Analysis API is running"})

@app.route('/api/prompt/analyze', methods=['POST'])
def analyze_prompt():
    """
    Analyze a prompt using Groq API
    Expected JSON format:
    {
        "prompt": "text to analyze",
        "provider": "groq", 
        "model": "llama-3.1-70b-versatile",
        "api_key": "your_api_key",
        "style": "comprehensive"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['prompt', 'api_key']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing or empty required field: {field}'}), 400
        
        prompt_text = str(data['prompt']).strip()
        api_key = str(data['api_key']).strip()
        provider = data.get('provider', 'groq')
        model = data.get('model', 'llama-3.3-70b-versatile')
        style = data.get('style', 'comprehensive')
        
        # Validate API key format (basic check)
        if len(api_key) < 10:
            return jsonify({'error': 'API key appears to be invalid (too short)'}), 400
        
        # Validate prompt length
        if len(prompt_text) < 3:
            return jsonify({'error': 'Prompt is too short. Please provide a meaningful prompt to analyze.'}), 400
        
        # Use default temperature and max_tokens
        temperature = 0.7
        max_tokens = 1000
        
        # Analyze the prompt using our app_util function with multi-provider support
        score, analysis_result = prompt_analysis(
            query=prompt_text,
            api_key=api_key,
            temp=temperature,
            max_token=max_tokens,
            provider=provider,
            model=model,
            style=style
        )
        
        # Check if there was an error
        if score == "Error":
            # Provide more specific error messages based on common issues
            error_msg = str(analysis_result)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                return jsonify({'error': 'Invalid API key. Please check your Groq API key and try again.'}), 401
            elif "rate limit" in error_msg.lower():
                return jsonify({'error': 'Rate limit exceeded. Please wait and try again.'}), 429
            elif "quota" in error_msg.lower():
                return jsonify({'error': 'API quota exceeded. Please check your Groq account.'}), 403
            else:
                return jsonify({'error': f'Analysis failed: {error_msg}'}), 500
        
        # Extract comprehensive analysis data
        if isinstance(analysis_result, dict):
            # New comprehensive format
            overall_score = analysis_result.get("overall_score", 75)
            detailed_scores = analysis_result.get("detailed_scores", {})
            strengths_list = analysis_result.get("strengths", ["Basic functionality"])
            weaknesses_list = analysis_result.get("weaknesses", ["Needs improvement"])
            improvements_list = analysis_result.get("improvements", ["Add more detail"])
            new_prompt = analysis_result.get("new_prompt", prompt_text)
            reasoning = analysis_result.get("reasoning", "Analysis completed")
            
            # Convert lists to formatted strings
            strengths = "• " + "\n• ".join(strengths_list) if strengths_list else "Basic prompt structure identified"
            weaknesses = "• " + "\n• ".join(weaknesses_list) if weaknesses_list else "Areas for improvement identified"
            improvements = "• " + "\n• ".join(improvements_list) if improvements_list else "General enhancements suggested"
            
            # Create comprehensive scores structure
            scores = {
                'overall': overall_score,
                'clarity': detailed_scores.get('clarity', overall_score - 5),
                'context': detailed_scores.get('context', overall_score - 3),
                'structure': detailed_scores.get('structure', overall_score - 4),
                'role': detailed_scores.get('role', overall_score - 8),
                'constraints': detailed_scores.get('constraints', overall_score - 12),
                'advanced': detailed_scores.get('advanced', overall_score - 15),
                # Legacy compatibility
                'specificity': detailed_scores.get('clarity', overall_score - 5),
                'effectiveness': detailed_scores.get('advanced', overall_score - 10)
            }
            
        else:
            # Legacy format fallback
            try:
                overall_score = int(score) if isinstance(score, (int, float)) else int(str(score).strip())
                overall_score = max(1, min(100, overall_score))
            except (ValueError, TypeError):
                overall_score = 75
            
            new_prompt = analysis_result if isinstance(analysis_result, str) else prompt_text
            
            # Generate basic feedback
            if overall_score >= 85:
                strengths = "• Excellent prompt with clear structure\n• Specific instructions provided\n• Good context and detail level"
                weaknesses = "• Very minor improvements possible\n• Could enhance precision in some areas"
            elif overall_score >= 70:
                strengths = "• Good prompt foundation\n• Clear intent and structure\n• Context generally well provided"
                weaknesses = "• Could benefit from more specific instructions\n• Clearer constraints needed\n• Some areas need better clarification"
            elif overall_score >= 50:
                strengths = "• Basic prompt structure present\n• Some clear elements identified\n• General direction provided"
                weaknesses = "• Needs improvement in specificity\n• Context and instruction clarity lacking\n• More detailed requirements needed"
            else:
                strengths = "• Has basic elements to build upon\n• Shows attempt at structure"
                weaknesses = "• Requires significant improvement in clarity\n• Needs better specificity and context\n• Instructions should be much more detailed"
            
            improvements = "• Add more specific instructions\n• Provide better context\n• Improve overall structure"
            reasoning = "Basic analysis completed"
            
            # Create scores structure
            import random
            random.seed(hash(prompt_text) % 1000)
            base_variance = 5
            scores = {
                'overall': overall_score,
                'clarity': max(1, min(100, overall_score + random.randint(-base_variance, base_variance))),
                'context': max(1, min(100, overall_score + random.randint(-base_variance, base_variance))),
                'structure': max(1, min(100, overall_score + random.randint(-base_variance, base_variance))),
                'role': max(1, min(100, overall_score + random.randint(-base_variance*2, base_variance))),
                'constraints': max(1, min(100, overall_score + random.randint(-base_variance*2, base_variance))),
                'advanced': max(1, min(100, overall_score + random.randint(-base_variance*3, base_variance))),
                'specificity': max(1, min(100, overall_score + random.randint(-base_variance, base_variance))),
                'effectiveness': max(1, min(100, overall_score + random.randint(-base_variance, base_variance)))
            }
        
        # Ensure new_prompt is a string
        if not isinstance(new_prompt, str):
            new_prompt = str(new_prompt)
        
        result = {
            'success': True,
            'scores': scores,
            'feedback': {
                'strengths': strengths,
                'weaknesses': weaknesses
            },
            'improved_prompt': new_prompt,
            'original_prompt': prompt_text,
            'provider': provider,
            'model': model,
            'analysis_style': style
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in analyze_prompt: {error_details}")  # For debugging
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/')
def serve_index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get available AI providers"""
    try:
        from app_util import get_supported_providers, PROVIDER_CONFIGS
        providers = []
        for provider_id in get_supported_providers():
            config = PROVIDER_CONFIGS[provider_id]
            providers.append({
                'id': provider_id,
                'name': config['name'],
                'models': config['models'],
                'default_model': config['default_model']
            })
        return jsonify({'providers': providers})
    except Exception as e:
        return jsonify({'error': f'Failed to get providers: {str(e)}'}), 500

@app.route('/api/models/<provider>', methods=['GET'])
def get_models(provider):
    """Get available models for a specific provider"""
    try:
        from app_util import get_provider_models, PROVIDER_CONFIGS
        models = get_provider_models(provider)
        if not models:
            return jsonify({'error': f'Provider {provider} not found'}), 404
        
        return jsonify({
            'provider': provider,
            'models': models,
            'default_model': PROVIDER_CONFIGS[provider]['default_model']
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get models: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Prompt Analyzer...")
    print("📍 Access the application at: http://localhost:5000")
    print("🔧 Make sure you have your Groq API key ready!")
    app.run(host='0.0.0.0', port=5000, debug=True)

