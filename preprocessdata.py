import pandas as pd
import json
import time
from llama_cpp import Llama

from openai import OpenAI, OpenAIError

API_KEY = "AIzaSyD_TaTYsYLspqyRR-O_iKtFj0XmYpy5tXM"
llama_key = "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"
groq_key = "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"
openrouter_key = "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"

def getValueMeaning(value):
    if(str(value).capitalize() == str("N")):
        return "No"
    elif(str(value).capitalize() == str("Y")):
        return "Yes"
    elif(type(value) == int):
        return str(value)+"/10"
    else:
        return str(value)

def load_model(model_path):
    """
    Load a locally stored GGUF Llama model using llama-cpp-python.
    """
    print("Loading model...")
    model = Llama(model_path=model_path, n_ctx=128)  # Adjust context window size if necessary
    print("Model loaded successfully!")
    return model

def rephrase_sentence(model, prompt, max_tokens=200, temperature=0.5, top_p=0.95):
    """
    Generate a rephrased sentence using the loaded model.
    """
    print("Generating rephrased output...")
    response = model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["\n"]  # Stop generation at the end of a sentence or line
    )
    # Extract and return the response text
    return response["choices"][0]["text"].strip()

def getRephrased2(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )

    while True:  # Loop until the request succeeds or another error occurs
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "openrouter.ai",
                    "X-Title": "openrouter.ai",
                },
                model="meta-llama/llama-3-8b-instruct:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return completion.choices[0].message.content

        except OpenAIError as e:
            if "rate limit" in str(e).lower():  # Check if the error is a rate limit error
                print("Rate limit exceeded. Retrying in 1 minute and 10 seconds...")
                time.sleep(70)  # Wait 1 minute and 10 seconds
            else:
                # If a different error occurs, raise it to avoid an infinite loop
                raise e

def remove_whitespace_and_newlines(text):
    # Removes all spaces and newline characters
    return text.replace(" ", "").replace("\n", "")

def getRephrased(text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key="+API_KEY
    # prompt = " rephrase the questions in the given text starting and ending with --- like the given example. Example: 'Patient age is 19, sex is F. Antecedents:Do you drink alcohol excessively or do you have an addiction to alcohol? Y ; Do you have chronic kidney failure? Y' Rephrased:' Patient age is 19, sex is F. Antecedents: I drink alcohol excessively or I have an addiction to alcohol.; I have a chronic kidney failure'.Keep all other text the same. --- {} ---"

    prompt_symptom = " rephrase the questions in the given text starting and ending with --- like the given example. Example: 'Patient age is 19, sex is F. Antecedents: Do you drink alcohol excessively or do you have an addiction to alcohol? Yes ; Have I had one or several flare ups of chronic obstructive pulmonary disease (COPD) in the past year? Yes; Do you have chronic kidney failure? No; Symptoms: Do you have pain in your stomach? Yes; Rephrased: Patient age is 19, sex is Female. Antecedents: I drink alcohol excessively or I have an addiction to alcohol; I had flare ups of chronic obstructive pulmonary disease (COPD) in the past year; I have a chronic kidney failure; Symptoms: I have pain in my stomach. Keep all other text the same return the repose between --- do not add --- in between. --- {} ---"

    prompt = prompt_symptom.format(text)
    response = getRephrased2(prompt)
    print(response)
    return response

def create_text_representation(row, evidence_dict):
    # Gather patient information
    age = row['AGE']
    sex = "Male" if row['SEX'] == 'M' else "Female" if row['SEX'] == 'F' else " "
    pathology = row['PATHOLOGY']
    initial_evidence = row['INITIAL_EVIDENCE']
    evidences = eval(row['EVIDENCES'])
    evidences = [initial_evidence] + evidences
    data = eval(row['DIFFERENTIAL_DIAGNOSIS'])
    diseases = [item[0] for item in data]
    diseases = ', '.join(diseases)
    # Build patient description
    description = f"Patient age is {age}, sex is {sex}. "
    # Add detailed symptoms and antecedents
    symptom_texts = []
    antecedents = []
    for evidence_code in evidences:
        # Separate multi-choice evidence by value
        if "_@_" in evidence_code:
            evidence, value = evidence_code.split('_@_')
            evidence_text = evidence_dict[evidence]['question_en']
            value_text = evidence_dict[evidence]['value_meaning'].get(value)
            value_text = value_text['en'] if value_text is not None else value
            rephrased_text = evidence_text + ' ' + getValueMeaning(value_text) + ' '
            if evidence_dict[evidence]['is_antecedent']:
                antecedents.append(rephrased_text)
            else:
                symptom_texts.append(rephrased_text)
        else:
            rephrased_text = evidence_dict[evidence_code]['question_en']+' Yes '
            if evidence_dict[evidence_code]['is_antecedent']:
                antecedents.append(rephrased_text)
            else:
                symptom_texts.append(rephrased_text)

    description += " Antecedents: " + "; ".join(antecedents) + ". Symptoms: " + "; ".join(symptom_texts) + ". "
    label = diseases
    return (description, label, pathology)

def process():
    with open('./22687585/release_evidences.json', 'r') as file:
        evidence = json.load(file)
    
    df = pd.read_csv('./22687585/release_train_patients/sampled_train_combined_data_100k.csv')
    with open('./22687585/release_train_patients/sampled_trained_data_100k.json', 'w') as data:
        data.write("[")
        counter = 0  # Initialize counter to keep track of entries written

        for _, row in df.iterrows():
            counter+=1
            string = create_text_representation(row, evidence)
            fileWrite = {
                "instruction": "Provide Diagnosis",
                "input": string[0],
                "output": f" Differential diagnosis is: {string[1]} and the Disease can be {string[2]} "
            }

            # Convert to JSON string
            json_output = json.dumps(fileWrite, ensure_ascii=False)

            # Write to the file
            clean_output = str(json_output)
            data.write(clean_output + ",\n")
        data.write("]")

def preProcessSelfDiagnosis(prompt_file):
    with open(prompt_file, 'r') as prompt_file, open("./self_diag", 'w') as self_diag_file:
        lines = prompt_file.readlines()
        counter = 0
        for line in lines:
            response = getRephrased(line)
            counter+=1
            if(counter == 19):
                time.sleep(62)
                counter = 0

            text = response

            print(response)

            cleaned_text = text.split('---');

            print(cleaned_text)


            self_diag_file.write(cleaned_text)

def clean_description(description, evidences):
    tmp = description.split("---")
    print(tmp)
    retStr = " "
    for s in tmp:
        if "Patient" in s and "I " in s:
            retStr = s
    if len(retStr) == 1:
        retStr = ''.join(tmp)
    retStr = retStr.replace("\n", "")
    retStr = retStr.replace("Here is the rephrased text starting and ending with", '')
    retStr = retStr.replace("Here is the rephrased text:", '')
    retStr = retStr.replace("Here is the rephrased text", '')

    for evidence in evidences:
        if evidences[evidence]['question_en'] in retStr:
            retStr = retStr.replace(evidences[evidence]['question_en'], '')
    return retStr




if __name__ == '__main__':
    # model = load_model("./HindiMedLLM/models/Llama-3.2-1B-Instruct-IQ3_M.gguf")
    process()
    # preProcessSelfDiagnosis('C:/Users/urani/Documents/MSCS/CS6220 BDA/Project/22687585/release_train_patients/training_data_2.txt')
    # promtp = "rephrase the questions in the given text starting and ending with --- like the given example. Example: 'Patient age is 19, sex is F. Antecedents:Do you drink alcohol excessively or do you have an addiction to alcohol? Y ; Do you have chronic kidney failure? N, Symptoms: Do you have pain in your stomach? Y Disease can be HIV (initial infection)' Rephrased:' Patient age is 19, sex is Female. Antecedents: I drink alcohol excessively or I have an addiction to alcohol, I have a chronic kidney failure Symptoms: I have pain in my stomach Disease can be HIV (initial infection)'.Keep all other text the same. --- Patient age is 21, sex is M. Antecedents:Have you ever had a sexually transmitted infection? Y ; Have you had unprotected sex with more than one partner in the last 6 months? Y ; Have you had sexual intercourse with an HIV-positive partner in the past 12 months? Y ; Have you traveled out of the country in the last 4 weeks?: N. Symptoms: Have you had significantly increased sweating? Y ; Do you have swollen or painful lymph nodes? Y ; Have you had significantly increased sweating? Y ; Have you had diarrhea or an increase in stool frequency? Y ; Do you have pain somewhere, related to your reason for consulting? Y ; Characterize your pain:: exhausting; Do you feel pain somewhere?: top of the head; Do you feel pain somewhere?: temple(R); Do you feel pain somewhere?: temple(L); How intense is the pain?: 7; Does the pain radiate to another location?: nowhere; How precisely is the pain located?: 7; How fast did the pain appear?: 2; Do you have a fever (either felt or measured with a thermometer)? Y ; Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for? Y ; What color is the rash?: pale; Do your lesions peel off?: N; Is the rash swollen?: 0; Where is the affected region located?: lower gum; Where is the affected region located?: upper gum; Where is the affected region located?: labia majora(R); Where is the affected region located?: internal cheek(R); Where is the affected region located?: internal cheek(L); How intense is the pain caused by the rash?: 6; Is the lesion (or are the lesions) larger than 1cm?: Y; How severe is the itching?: 0; Are you feeling nauseous or do you feel like vomiting? Y ; Have you had an involuntary weight loss over the last 3 months? Y .  Differential diagnosis is :HIV (initial infection), Chagas, Scombroid food poisoning, Sarcoidosis Disease can be HIV (initial infection)---"
    # print(getRephrased2(promtp).split("---"))