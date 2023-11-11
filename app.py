import os
import spacy
import pandas as pd
from flask import Flask, render_template, request
from fuzzywuzzy import fuzz
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Get the path to the wwwroot directory
wwwroot_path = os.path.abspath(os.path.dirname(__file__))

# Construct the file path to the qna.csv file
csv_file_path = os.path.join(wwwroot_path, "qna.csv")

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Initialize an empty dictionary for the dataset
dataset = {}

# Function to load your custom dataset from a CSV file
def load_custom_dataset(filename):
    try:
        # Read the CSV file with questions and answers
        df = pd.read_csv(filename)
        
        # Assuming your CSV file has columns named "ques" and "answer"
        for index, row in df.iterrows():
            question = row["ques"].strip()  # Remove leading/trailing spaces
            answer = row["answer"].strip()  # Remove leading/trailing spaces
            dataset[question] = answer
    except Exception as e:
        print(f"Error loading custom dataset: {e}")

# Hardcode your questions and answers here
qa_data = {
    "hello": "Hi",
    "how are you?": "Good, you?",
    "how is it going?": "Great",
    "good": "Same here",
    "great": "That is good to hear",
    "what color is the sky": "Blue",
    "goodbye": "Goodbye",
    "what is your name": "My name is Chatbot",
    "can you tell me your name?": "My name is Chatbot",
    "what information is required to register for the legal platform": "The following information is required to register for the legal platform: Name, email address, preferred language. Users may also be asked to create a password and provide additional information, such as their location or area of interest.",
    "how is user data stored and secured": "User data for a legal platform is typically stored and secured using a variety of methods, including: Encryption, Access control, Auditing, Regular backups.",
    "what are the benefits of registering for the legal platform": "There are many benefits to registering for a legal platform, including: Access to legal information and resources, convenience and ease of use, affordability.",
    "what essential links are available on the homepage": "The essential links available on the homepage of a legal platform may vary depending on the specific platform, but they typically include: Know-Your-Rights (KYR) framework, Legal Events Calendar, Case-Based Learning of Legal Issues, Legal Aid Providers.",
    "how is the KYR framework organized": "It can be organized by the hierarchical structure, and it may also be organized by legal rights, legal topics, and legal cases.",
    "what types of legal information can be found in the KYR framework": "The KYR framework in a legal platform can provide users with a wide range of legal information, including: Laws and regulations, Case law, Legal articles, Legal forms.",
    "what is KYR": "KYR typically stands for Know Your Rights. It refers to the awareness and understanding of one's legal rights and protections in various situations, such as interactions with law enforcement, employment, housing, and more. Knowing your rights is essential for individuals to protect themselves and advocate for their interests within the boundaries of the law.",
    "how can users search for specific legal information in the KYR framework": "To search for specific legal information in the KYR framework, users can browse it and can use some filters.",
    "how can users obtain assistance from the chatbot": "To obtain assistance from the chatbot if they have questions or need clarifications about the legal content, users can access it from the homepage and then they can type their questions or requests, and then the chatbot will use its knowledge to try to answer the user's question or will ask the user for additional information. If the user is not satisfied with the chatbot's response, they can ask follow-up questions.",
    "what types of legal events are listed in the calendar": "The types of legal events that are listed in the calendar are: Court hearings, which is the dates and times of upcoming court hearings. Document filing deadlines, which list the deadlines for filing important legal documents, such as lawsuits, appeals, and tax returns. Bar association meetings and events, which may list the dates and times of upcoming bar association meetings and events.",
    "what variety of legal topics are covered in the case-based learning materials": "Constitutional law, Criminal law, Civil law, Family law, Business law, White collar crime, Immigration law, Environmental law.",
    "what types of legal aid providers are listed on the page": "Legal aid providers listed on a legal platform may include nonprofit legal aid organizations, law school clinics, pro bono programs, government legal aid programs, public defenders, and private attorneys. Some legal platforms may also list self-help resources, online legal services, and legal technology tools.",
    "how can users search for legal aid providers by location, area of law, or other criteria": "Users can search for legal aid providers on a legal platform by location, area of law, or other criteria by using the platform's search bar and filters. For example, a user can enter their zip code and 'family law' to find legal aid providers in their area who specialize in family law. Other common search criteria include income eligibility, language spoken, and availability of evening and weekend appointments.",
    "what information is provided about each legal aid provider": "The information provided about each legal aid provider on a legal platform may include the name of the organization or provider, contact information, website address. This information can help users choose a legal aid provider that is right for their needs."
}

# Add the hardcoded Q&A data to the dataset
dataset.update(qa_data)

# Define a function to get chatbot responses
def chatbot_response(user_input):
    user_input = user_input.lower()
    
    # Tokenize the user input using NLTK
    user_input_tokens = word_tokenize(user_input)
    
    # Initialize variables to track the best match
    best_match_question = None
    best_match_score = 0
    
    # Loop through the dataset questions and calculate fuzzy match scores
    for question in dataset.keys():
        similarity_score = fuzz.ratio(user_input, question.lower())
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_question = question
    
    # Check if the best match score is above a certain threshold
    if best_match_score > 51:  # Adjust the threshold as needed
        return dataset[best_match_question]
    else:
        # Use spaCy to extract named entities (e.g., names)
        user_input_doc = nlp(user_input)
        for ent in user_input_doc.ents:
            if ent.label_ == "PERSON":
                return f"My name is {ent.text}."
        
        return "I'm not sure how to answer that."

@app.route("/")
def home():
    return render_template("index.html", qa_data=dataset,csv_file_path=csv_file_path)
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")
    response = chatbot_response(user_input)
    return render_template("index.html", user_input=user_input, response=response, qa_data=dataset)

if __name__ == "__main__":
    load_custom_dataset(csv_file_path)
    app.run(debug=True)
