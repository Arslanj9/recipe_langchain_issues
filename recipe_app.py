from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Verify key is loaded
print("HF Token exists:", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))

# -------------------------------
# 1. Initialize Hugging Face LLM
# -------------------------------
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",   # free instruction-tuned model
    temperature=0.7,
    max_new_tokens=512
)

print("✅ Step 1: Hugging Face LLM initialized")

# -------------------------------
# 2. Define Prompts
# -------------------------------
recipe_prompt = PromptTemplate.from_template(
    "Generate a recipe for {dish} in a {tone} tone. "
    "Include ingredients and step-by-step instructions. "
    "Do not exceed 200 words."
)

nutrition_prompt = PromptTemplate.from_template(
    "Provide a nutritional breakdown of {dish}. "
    "Respond in JSON with fields: "
    "{{'calories': '...', 'protein': '...', 'carbs': '...', 'fat': '...', 'health_tips': '...'}}"
)

print("✅ Step 2: Prompts created")

# -------------------------------
# 3. Test LLM connection
# -------------------------------
try:
    test_output = llm.invoke("Hello! Reply with exactly 5 words.")
    print("✅ Step 3: LLM test successful:", test_output)
except Exception as e:
    print("❌ Step 3: LLM test failed:", e)

# -------------------------------
# 4. Preview formatted prompts
# -------------------------------
try:
    preview_recipe = recipe_prompt.format(dish="Chicken Biryani", tone="funny")
    preview_nutrition = nutrition_prompt.format(dish="Chicken Biryani")
    print("✅ Step 4: Prompt formatting successful")
    print("Recipe prompt preview:\n", preview_recipe)
    print("Nutrition prompt preview:\n", preview_nutrition)
except Exception as e:
    print("❌ Step 4: Prompt formatting failed:", e)

# -------------------------------
# 5. Define Chains
# -------------------------------
recipe_chain = recipe_prompt | llm | StrOutputParser()
nutrition_chain = nutrition_prompt | llm | StrOutputParser()
print("✅ Step 5: Chains defined")

# -------------------------------
# 6. Test individual chains
# -------------------------------
try:
    recipe_result = recipe_chain.invoke({"dish": "Chicken Biryani", "tone": "funny"})
    print("\n✅ Step 6a: Recipe chain output:\n", recipe_result[:300], "...")
except Exception as e:
    print("❌ Step 6a: Recipe chain failed:", e)

try:
    nutrition_result = nutrition_chain.invoke({"dish": "Chicken Biryani", "tone": "funny"})
    print("\n✅ Step 6b: Nutrition chain output:\n", nutrition_result[:300], "...")
except Exception as e:
    print("❌ Step 6b: Nutrition chain failed:", e)

# -------------------------------
# 7. Parallel execution
# -------------------------------
parallel_chain = RunnableParallel(
    recipe=recipe_chain,
    nutrition=nutrition_chain
)

try:
    final_result = parallel_chain.invoke({
        "dish": "Chicken Biryani",
        "tone": "funny"
    })
    print("\n✅ Step 7: Parallel execution successful")
    print("Final result:", final_result)
except Exception as e:
    print("❌ Step 7: Parallel execution failed:", e)
