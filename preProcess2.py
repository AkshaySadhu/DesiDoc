import requests
import json

if __name__ == '__main__':

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyD_TaTYsYLspqyRR-O_iKtFj0XmYpy5tXM"

    payload = json.dumps({
      "contents": [
        {
          "parts": [
            {
              "text": " rephrase the questions in the given text starting and ending with --- like the given example. Example: 'Do you have chronic kidney failure? Y' Rephrased:'I have a chronic kidney failure'.Keep all other text the same. ---Patient age is 39, sex is M. Antecedents:Do you have a poor diet? Y ; Have you ever had a diagnosis of anemia? Y ; Do you have chronic kidney failure? Y ; Are you taking any new oral anticoagulants ((NOACs)? Y ; Have you traveled out of the country in the last 4 weeks?: Central America; Is your BMI less than 18.5, or are you underweight? Y . Symptoms: Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? Y ; Do you have pain somewhere, related to your reason for consulting? Y ; Characterize your pain:: tugging; Characterize your pain:: a cramp; Do you feel pain somewhere?: forehead; Do you feel pain somewhere?: temple(L); How intense is the pain?: 1; Does the pain radiate to another location?: nowhere; How precisely is the pain located?: 3; How fast did the pain appear?: 4; Do you feel lightheaded and dizzy or do you feel like you are about to faint? Y ; Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? Y ; Have you recently had stools that were black (like coal)? Y ; Is your skin much paler than usual? Y .  Differential diagnosis is :Anemia, Anaphylaxis, Chagas, Cluster headache, Scombroid food poisoning Disease can be Anemia ---"
            }
          ]
        }
      ]
    })
    headers = {
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
