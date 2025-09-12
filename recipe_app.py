from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# -------------------------------
# 0. Load environment variables
# -------------------------------
load_dotenv()
print("Google API Key exists:", bool(os.getenv("GOOGLE_API_KEY")))

# -------------------------------
# 1. Initialize Google Gemini LLM
# -------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # fast & free tier; use "gemini-1.5-pro" for stronger model
    temperature=0.7,
    max_output_tokens=300
)
print("✅ Step 1: Google Gemini LLM initialized")

# -------------------------------
# 2. Define a single prompt
# -------------------------------
recipe_prompt = PromptTemplate.from_template(
    "Generate a recipe for {dish} in a {tone} tone."
)
print("✅ Step 2: Recipe prompt created")

# -------------------------------
# 3. Test LLM connection
# -------------------------------
try:
    test_output = llm.invoke("Hello! Reply with exactly 5 words.")
    print("✅ Step 3: LLM test successful:", test_output.content)
except Exception as e:
    print("❌ Step 3: LLM test failed:", e)

# -------------------------------
# 4. Format and directly invoke LLM
# -------------------------------
try:
    formatted_prompt = recipe_prompt.format(dish="Chicken Biryani", tone="funny")
    print("✅ Step 4: Prompt formatting successful")
    print("Formatted recipe prompt:\n", formatted_prompt)

    recipe_result = llm.invoke(formatted_prompt)
    print("\n✅ Step 5: LLM direct invocation successful")
    print("Recipe result:\n", recipe_result.content)
except Exception as e:
    print("❌ Step 5: Direct LLM invocation failed:", e)
