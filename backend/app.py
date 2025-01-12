import os
import requests
import zipfile
from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://custom-llm-lvvzz6gni-suns-projects-fbf4e3bd.vercel.app"}})


MODEL_DIR = "models/fine_tuned_model"

S3_URL = "https://sunthecoder-models.s3.us-east-2.amazonaws.com/fine_tuned_model.zip?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMiJHMEUCIDXROUVbzxlQ9x31y%2BIbFB4NmbrTatK1Mhrfi1co78BoAiEAxBX1sYxJLcTdrrN18G%2BFNV8judStbnA7weuidjQEVUkq0AMIyv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw4OTE2MTI1ODcyMzkiDAlKZBly5yCW5hBLAiqkA4z4QeXRUDfpGnxFhX1li38wVXtl9svkuagS6oWRdkL1HRp8beu0BXnOK%2BUAY%2BcZe4Entd%2FQy0%2FriYo3iAF7meDzpZmhuscE6qlelf7X42bCVnXqV5791I%2BThPdJoc7X2EqTv7SmMr5FsZgkQeJLKMXpexwXrPPOE8LXtA4wX7ZXxOs9PqfzCEzjri5ypA2pf9psHLu2JnJtkssvoiVGn83XcojgQL%2FFc9GEVChaWqVBQOL0fG%2F5Y8GMLL7O9IekXvwdvtEXvGbLTp6yxqDM2kbAq4FZssIeT00mpIHeZluMErdH3n4XPjTG9MRJdasOWVvmL0oMHo2UWvIgPjudabiS5oOVB5qkDgg9BS8bMAmNWVABmyPzqi2uFH2IIinhvVdD1LZ5TPsqvDaENOGfbEbfxuPbxbkcCMUXrrSWcfl2rnIyr3Y0moIQbY9EOQIVgGUEXZMch%2BXwU%2Bbrfx7oDtgVypuGZkrRFlZvbxHNVhtjOZZs2PCZYsvcDL7el6Hrn2i2Dywyxk%2FRigaTkBvmYrM7OUNk13A6tXY93D3e%2B%2B%2FlTHGOzDD7k4y8BjrkAjAGqDPIv77SpH3et%2BeL3BstLxSLcp5I5hboY7ExdEyykvsOZZmFR2zJs3%2FtC0YHCK%2FERcHVX8zlplXeBuV6Vhp5YrZkBxMe%2FDjOhnjznGabp5h125r16kFQMWPWG0TQ2G2Lw9jksAp%2F%2FwCuDcqxuyhOhO075vGj%2B4xAC48Rp%2FjjQWQIse72wprt3qq8ZBldDiD3DliEUE3fEV4HkSVyjpl8Tlr%2F1ReHU7y0Xmkxcr4RiSANratji7W%2BodgcXS2jWHA8nSpyvVEFJboKG0TqgANqLtetbMava82iwGXfnP8EezDFzkN%2BFRnYd5y8mKK1R7xP9eu%2BRONL2yvYCZ9H2QggCi0o8QtZehVKMkJnJqHKwxzL1u%2FkondTs5L8xU%2BN9QCjznQd8nkYmPxaLWAK1%2FI6EnxgM8ChIrHoYI%2FDUgkcazRxxZ6gSHK6Po%2FzWZkNzDqBVQtAOxb8m5nsfyoM0V68g2so&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA47GCAMTT42PAHXDR%2F20250112%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20250112T004312Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=f58285fe043e9d558fd287531e60ebd55c869b38ef15ba64d4be3bea11c71b40"
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
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable if provided
    app.run(debug=True, host='0.0.0.0', port=port)

