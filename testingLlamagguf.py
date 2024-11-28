import llama_cpp

# Load the fine-tuned model and tokenizer
model_path = "./HindiMedLLM/models/Llama-3.2-1B-Instruct-IQ3_M.gguf"

# Load the llama-cpp model
print("Loading the model...")
llama_model = llama_cpp.Llama(model_path=model_path)
print("Model loaded successfully!")

# Function to generate responses using llama-cpp
def generate_response(prompt, max_tokens=1024, temperature=0.7, top_p=0.95):
    """Generate a response for a given prompt using llama-cpp."""
    response = llama_model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["\n"],  # Stop generation at a new line
    )
    return response["choices"][0]["text"]

# Test prompts
print("\nTesting the model with custom prompts...")

test_prompts = [
    "Instruction: Provide Diagnosis Patient age is 65, sex is M.  Antecedents: Do you have a known issue with one of your heart valves? Y ; Do you have severe Chronic Obstructive Pulmonary Disease (COPD)? Y ; Do you have diabetes? Y ; Do you have high blood pressure or do you take medications to treat high blood pressure? Y ; Do you have a known heart defect? Y ; Have you traveled out of the country in the last 4 weeks? N . Symptoms: Do you feel slightly dizzy or lightheaded? Y ; Are you experiencing shortness of breath or difficulty breathing in a significant way? Y ; Do you feel slightly dizzy or lightheaded? Y ; Do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations? Y ; Do you feel your heart is beating very irregularly or in a disorganized pattern? Y ; Do you have symptoms that are increased with physical exertion but alleviated with rest? Y .",
    "Instruction: Provide Diagnosis Patient age is 52, sex is F.  Antecedents: Have you traveled out of the country in the last 4 weeks? N . Symptoms: Are your symptoms worse when lying down and alleviated while sitting up? Y ; Do you have pain somewhere, related to your reason for consulting? Y ; Characterize your pain: a knife stroke ; Do you feel pain somewhere? side of the chest(R) ; Do you feel pain somewhere? side of the chest(L) ; Do you feel pain somewhere? breast(R) ; Do you feel pain somewhere? breast(L) ; How intense is the pain? 8 ; Does the pain radiate to another location? nowhere ; How precisely is the pain located? 3 ; How fast did the pain appear? 3 ; Do you feel out of breath with minimal physical effort? Y ; Are you experiencing shortness of breath or difficulty breathing in a significant way? Y ; Are your symptoms worse when lying down and alleviated while sitting up? Y ; Do you have symptoms that are increased with physical exertion but alleviated with rest? Y .",
    "Instruction: Provide Diagnosis Patient age is 32, sex is M.  Antecedents: Have you recently had a viral infection? Y ; Have you ever had a pericarditis? Y ; Have you traveled out of the country in the last 4 weeks? N . Symptoms: Do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations? Y ; Do you have pain somewhere, related to your reason for consulting? Y ; Characterize your pain: a knife stroke ; Do you feel pain somewhere? lower chest ; Do you feel pain somewhere? upper chest ; Do you feel pain somewhere? breast(R) ; Do you feel pain somewhere? breast(L) ; Do you feel pain somewhere? epigastric ; How intense is the pain? 6 ; Does the pain radiate to another location? thoracic spine ; Does the pain radiate to another location? posterior chest wall(R) ; Does the pain radiate to another location? posterior chest wall(L) ; How precisely is the pain located? 5 ; How fast did the pain appear? 6 ; Are you experiencing shortness of breath or difficulty breathing in a significant way? Y ; Do you feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations? Y .",
]

for i, prompt in enumerate(test_prompts):
    print(f"\nPrompt {i+1}:\n{prompt}")
    response = generate_response(prompt)
    print(f"Response {i+1}:\n{response}")
