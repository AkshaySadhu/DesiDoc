import csv
import json
import time
import re

from openai import OpenAI, OpenAIError

API_KEY = "AIzaSyD_TaTYsYLspqyRR-O_iKtFj0XmYpy5tXM"
llama_key = "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"


def getRephrased2(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=llama_key,
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

    prompt_symptom = " rephrase the text in between #### based on the following example \"\"input: Did you lose consciousness? Y  Ouput: I lost consciousness\"\"  give the output in a single line between --- without any prefix or suffix. #### {} #### "

    prompt = prompt_symptom.format(text)
    response = getRephrased2(prompt)
    print(response)
    print(response)
    return response

def create_text_representation(row, evidence_dict):
    # Gather patient information
    age = row['AGE']
    sex = row['SEX']
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
            rephrased_text = evidence_text + value_text
            if evidence_dict[evidence]['is_antecedent']:
                antecedents.append(rephrased_text)
            else:
                symptom_texts.append(rephrased_text)
        else:
            rephrased_text = evidence_dict[evidence_code]['question_en']+' Y '
            if evidence_dict[evidence_code]['is_antecedent']:
                antecedents.append(rephrased_text)
            else:
                symptom_texts.append(rephrased_text)

    description += "Antecedents:" + "; ".join(antecedents) + ". Symptoms: " + "; ".join(symptom_texts) + ". "
    label = diseases

    return (description, label, pathology)

def process():
    with open('C:/Users/urani/Documents/MSCS/CS6220 BDA/Project/22687585/release_evidences.json', 'r') as file:
        evidence = json.load(file)
    
    with open('C:/Users/urani/Documents/MSCS/CS6220 BDA/Project/22687585/release_train_patients/release_train_patients', 'r') as file:
        reader = csv.DictReader(file)

        with open('C:/Users/urani/Documents/MSCS/CS6220 BDA/Project/22687585/release_train_patients/training_data_3.json', 'w') as data:
            data.write("[")
            for row in reader:
                string = create_text_representation(row, evidence)
                description = getRephrased(string[0]).replace('---', '').strip()
                fileWrite = dict()
                fileWrite['instruction'] = "Provide Diagnosis"
                fileWrite['input'] = description
                fileWrite['output'] = "Differential diagnosis is :" + string[1] + " and the Disease can be " + string[2]
                data.write(re.sub(r"Here is the rephrased text.*?(:|\n\n)", "", str(fileWrite).replace("'", "\"")) + ', \n')
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

            cleaned_text = text.replace('---', '').replace("Here is the rephrased text in a single line between :", "").strip()

            print(cleaned_text)

            self_diag_file.write(cleaned_text)




if __name__ == '__main__':
    process()
    # preProcessSelfDiagnosis('C:/Users/urani/Documents/MSCS/CS6220 BDA/Project/22687585/release_train_patients/training_data_2.txt')
    # promtp = "rephrase the questions in the given text starting and ending with --- like the given example. Example: 'Patient age is 19, sex is F. Antecedents:Do you drink alcohol excessively or do you have an addiction to alcohol? Y ; Do you have chronic kidney failure? N, Symptoms: Do you have pain in your stomach? Y Disease can be HIV (initial infection)' Rephrased:' Patient age is 19, sex is F. Antecedents: I drink alcohol excessively or I have an addiction to alcohol, I have a chronic kidney failure Symptoms: I have pain in my stomach Disease can be HIV (initial infection)'.Keep all other text the same. --- Patient age is 21, sex is M. Antecedents:Have you ever had a sexually transmitted infection? Y ; Have you had unprotected sex with more than one partner in the last 6 months? Y ; Have you had sexual intercourse with an HIV-positive partner in the past 12 months? Y ; Have you traveled out of the country in the last 4 weeks?: N. Symptoms: Have you had significantly increased sweating? Y ; Do you have swollen or painful lymph nodes? Y ; Have you had significantly increased sweating? Y ; Have you had diarrhea or an increase in stool frequency? Y ; Do you have pain somewhere, related to your reason for consulting? Y ; Characterize your pain:: exhausting; Do you feel pain somewhere?: top of the head; Do you feel pain somewhere?: temple(R); Do you feel pain somewhere?: temple(L); How intense is the pain?: 7; Does the pain radiate to another location?: nowhere; How precisely is the pain located?: 7; How fast did the pain appear?: 2; Do you have a fever (either felt or measured with a thermometer)? Y ; Do you have any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for? Y ; What color is the rash?: pale; Do your lesions peel off?: N; Is the rash swollen?: 0; Where is the affected region located?: lower gum; Where is the affected region located?: upper gum; Where is the affected region located?: labia majora(R); Where is the affected region located?: internal cheek(R); Where is the affected region located?: internal cheek(L); How intense is the pain caused by the rash?: 6; Is the lesion (or are the lesions) larger than 1cm?: Y; How severe is the itching?: 0; Are you feeling nauseous or do you feel like vomiting? Y ; Have you had an involuntary weight loss over the last 3 months? Y .  Differential diagnosis is :HIV (initial infection), Chagas, Scombroid food poisoning, Sarcoidosis Disease can be HIV (initial infection)---"
    # getRephrased2(promtp)