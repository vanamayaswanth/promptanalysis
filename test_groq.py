#!/usr/bin/env python3
"""
Test script to verify Groq API integration
"""
import os
from groq import Groq

def test_groq_connection():
    """Test basic Groq API connection"""
    print("🧪 Testing Groq API Connection...")
    
    # Get API key from environment or prompt
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        api_key = input("Enter your Groq API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    try:
        # Initialize client
        client = Groq(api_key=api_key)
        
        # Test basic completion
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "user", "content": "Say 'Hello, Groq API is working!' in exactly that format."}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        response = completion.choices[0].message.content
        print(f"✅ API Response: {response}")
        
        # Test our prompt analysis function
        from app_util import prompt_analysis
        
        test_prompt = "Write a story about a robot."
        print(f"\n🔍 Testing prompt analysis with: '{test_prompt}'")
        
        score, improved_prompt = prompt_analysis(
            query=test_prompt,
            api_key=api_key,
            temp=0.7,
            max_token=500
        )
        
        if score == "Error":
            print(f"❌ Prompt analysis failed: {improved_prompt}")
            return False
        
        print(f"✅ Analysis successful!")
        print(f"📊 Score: {score}")
        print(f"🔧 Improved prompt: {improved_prompt[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_groq_connection()
    if success:
        print("\n🎉 All tests passed! Your setup is working correctly.")
        print("🚀 You can now run: uv run main.py")
    else:
        print("\n💥 Tests failed. Please check your API key and try again.")