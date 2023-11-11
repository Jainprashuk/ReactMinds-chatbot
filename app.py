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
    "hello": "Hi",
"how are you?": "Good, you?",
"how is it going?": "Great",
"good": "Same here",
"great": "That is good to hear",
"what color is the sky": "Blue",
"goodbye": "Goodbye",
"what is your name": "My name is Chatbot",
"can you tell me your name?": "My name is Chatbot",
"what is React": "React is a JavaScript library for building user interfaces. It allows developers to create reusable UI components and build interactive, single-page applications.",
"React": "React is a JavaScript library for building user interfaces. It allows developers to create reusable UI components and build interactive, single-page applications.",
"what is JSX in React": "JSX (JavaScript XML) is a syntax extension for JavaScript that looks similar to XML or HTML. It allows you to write HTML elements and components in your JavaScript code, making it easier to build and visualize the UI.",
"what is the Virtual DOM in React": "The Virtual DOM is a concept in React that represents a lightweight copy of the actual DOM. React uses it to improve performance by updating only the parts of the DOM that have changed, rather than re-rendering the entire DOM.",
"what is the difference between state and props in React": "State is used for mutable data that can change over time and is managed by the component itself. Props (short for properties) are used to pass data from a parent component to a child component.",
"what is Redux, and why might you use it with React": "Redux is a state management library for JavaScript applications. It can be used with React to manage the state of an entire application in a predictable way, making it easier to debug and maintain large applications.",
"what is React Router": "React Router is a library that enables navigation and routing in a React application. It allows developers to define different routes and views for different parts of the application, enabling a better user experience in single-page applications.",
"how does React handle components": "React uses a component-based architecture. Components are independent, reusable pieces of code that can be composed to build complex UIs. They can have their own state and lifecycle methods.",
"what is the React component lifecycle": "React components go through three main phases: Mounting (component is created and inserted into the DOM), Updating (component re-renders when props or state change), and Unmounting (component is removed from the DOM).",
"what are controlled and uncontrolled components in React": "Controlled components are components whose behavior is controlled by React, typically by using state. Uncontrolled components are components that store their own state internally, outside of React.",
"how can you optimize performance in a React application": "Performance optimization in React can be achieved by using PureComponent, memoization, code splitting, optimizing the use of the Virtual DOM, and minimizing unnecessary re-renders.",
"what is the significance of the useEffect hook in React": "The useEffect hook in React is used for handling side effects in functional components. It allows you to perform tasks like data fetching, subscriptions, or manually changing the DOM after the component has rendered.",
"how does React handle forms": "React handles forms using controlled components, where form elements like input fields are controlled by state. Changes in the input fields trigger state updates, making it easy to access and manipulate form data.",
"what is the purpose of the useRef hook in React": "The useRef hook in React is used to create a mutable object called a ref. It is commonly used to access and interact with a DOM element or to persist values across renders without causing re-renders.",
"what is the difference between state and props in React": "State is used for mutable data that can change over time within a component, while props are used to pass data from a parent component to a child component.",
"how does React handle conditional rendering": "Conditional rendering in React is achieved using conditional statements, such as if statements or the ternary operator, based on the state or props of a component.",
"what is the purpose of the useMemo hook in React": "The useMemo hook in React is used to memoize the result of a computation, preventing unnecessary recalculations and improving performance in scenarios where expensive calculations are involved.",
"how can you handle events in React": "In React, events are handled using camelCase event names, and you can define event handlers as functions. Common events include onClick, onChange, and onSubmit.",
"what is the purpose of the useCallback hook in React": "The useCallback hook in React is used to memoize a callback function, preventing it from being recreated on each render and optimizing performance in scenarios where callbacks are dependencies for other hooks or components.",
"what is the role of keys in React lists": "Keys in React lists are used to uniquely identify elements and help React efficiently update and re-render components when the order or number of list items changes.",
"how can you pass data between components in React": "Data can be passed between components in React through props. Additionally, context API, state lifting, and Redux can be used for more complex scenarios.",
"what is the purpose of the useContext hook in React": "The useContext hook in React is used to consume values from the context API, providing a way for functional components to access values stored in a context provider.",
"how does React handle routing": "React handles routing using libraries like React Router, where different components are rendered based on the URL, allowing for the creation of single-page applications with multiple views.",
"what is the significance of the useReducer hook in React": "The useReducer hook in React is used to manage complex state logic by specifying how state transitions occur based on dispatched actions. It is often used as an alternative to useState in scenarios with more intricate state management.",
"how can you pass functions as props in React": "Functions can be passed as props in React by assigning a function to a prop in the parent component and then invoking that function from the child component when necessary.",
"what is the purpose of the useHistory hook in React Router": "The useHistory hook in React Router provides access to the history object, allowing programmatic navigation and manipulation of the browser's history stack.",
"how does React handle context": "React uses the context API to share values like themes or authentication status between components without passing props explicitly through every level of the component tree.",
"what is the purpose of the useState hook in React": "The useState hook in React is used to add state to functional components, allowing them to have dynamic behavior by updating and re-rendering based on changes to the state.",
"how does React handle forms": "React handles forms using controlled components, where form elements like input fields are controlled by state. Changes in the input fields trigger state updates, making it easy to access and manipulate form data.",
"what is the purpose of the useRef hook in React": "The useRef hook in React is used to create a mutable object called a ref. It is commonly used to access and interact with a DOM element or to persist values across renders without causing re-renders."
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
