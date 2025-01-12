import os
import requests
import zipfile
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

# Get absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models/fine_tuned_model")

S3_URL = "https://sunthecoder-models.s3.us-east-2.amazonaws.com/fine_tuned_model.zip?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMiJHMEUCIQC1mitNk46ni0ndRw%2BwjKnxctFrsbmWESJf4ZJdg4ZnwgIgVn8GR03hw%2BCXOiVD%2B5khK%2BvnbH%2BKGv%2BD2BEOPkumuS0q0AMIzP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw4OTE2MTI1ODcyMzkiDAPT0fmCoF%2FwGwKdECqkA8JNDdj1Ys5d%2FYfFlFTMxicc5kw7s3CYlucUgbvhBeNImcn8hoSd2Wr87HIjmNm3rIMfFBycyuA1YN0E1yZai6njMKCN6ItAU4yHSaG%2FTij9nXGX48e7%2BZacCAfkBxvbL2WGjQyWHjVeF2HlyKr8kynGjtT3EICpU9ju58PnvBJbTyaUzEb21PZgECboU%2BPA1CT4y9goUTp2zOhQ6FNAVnSEc4R9FjOURzQsOaJyzyHpvlDL%2FmePWLSvTjfR9XVbX492ZapWfXXetJjCdXkSTxoo%2FGylGDaU9nfqUEoi%2BuPLcaeMAIli3OedzgD%2BTcqUxB3tKRQZYE5tsCqSne87tU91dSYd7nSN23rarjHqip0E4YH071Fxj%2Bfs23g6w6KWdUZ%2BPwfM5WD20kl%2B%2FJJc%2B8V54zqRkhcCn4%2F%2BZdrA%2FveXlfRUPhFmG66XXBluiPzpuuP0HBVDt3tEmKd8tjqRaqxHu8PEq5UBzFN3ZTsdO5RLjRs8zdeKRC5BtlXWqsb3ohNQCsbT%2Bv87Zg%2BKoXhJO2UrHZJc%2BM0txdhKGhitcTB1OejaTzD7k4y8BjrkAmgpBQqd9dyDbRmmWl%2BDMjtvQOAF58NkJIehUBFrCAooezt74GOJjmK%2FFK5pWmZgFRbu439gLv4Rnnj3HSVDczK396KEYZRsLdf1RDYmoXmcXdt9RaHVslI74MZ6jR8zyhvB7fx1YN21ZfsVjJOEpdU0%2BxM80mlpUhR7SiLgMKqoeTRcgPZyBq5UnJPw55VdiH106kVm3ovPk7Y50JRc4ZPRbf%2BuW%2Fq6%2BXqMAVAllz%2Fe05Ac1%2F1YCysfFnYJZO7rzSGxz7MAD2QLGbVArk907%2FBS6eHfEt1%2F2WL75lL9xtGArecjiFWNSDwpne8jc3J9PF6oP0tknoC50ncHCcRzNttbPw%2F9UnyFdNHECrZG%2BdIC9j4t7qEIy46Yed7Y%2F81BtR4V059ZrAj0z%2Fm8bLdB5tX%2BZhr%2F1tSseGOaXqkRA38kBnlhGsFfQynhq0CchLiJFFEpaAf7WL26RW9zy7%2BB1UYw5Grb&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA47GCAMTTS2EHKDNX%2F20250112%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20250112T030924Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=73734dfc2ae86707796ca2e9eac03d48314a03fb2b23a08c740262b0fdb9c7cd"
ZIP_FILE = "fine_tuned_model.zip"

# Ensure the model is downloaded and ready
if not os.path.exists(MODEL_DIR):
    print("Downloading model from S3...")
    response = requests.get(S3_URL, stream=True)
    with open(ZIP_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    # Validate file size
    downloaded_size = os.path.getsize(ZIP_FILE)
    print(f"Downloaded file size: {downloaded_size} bytes")

    # Extract the model
    print("Extracting model...")
    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall("backend/models/")
        print("Model ready!")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
        os.remove(ZIP_FILE)
        exit(1)

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")
    
    # Tokenize with padding and attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Generate text
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        max_length=100, 
        num_return_sequences=1,
        temperature=0.7,  # Controls randomness; lower is more focused, higher is more random
        # top_k=50,         # Limits sampling to the top K tokens
        # top_p=0.9,        # Nucleus sampling (considering top P cumulative probability)
        pad_token_id=tokenizer.eos_token_id
)

    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
