# Install Libraries
! pip install swarmauri[full]==0.4.1 python-dotenv
! pip install gradio
! pip install python-dotenv
! pip install groq
# Importing necessary libraries
from dotenv import load_dotenv  # To load environment variables from a .env file
import os  # For interacting with the operating system (e.g., fetching environment variables)
import gradio as gr  # Gradio for creating user interfaces for machine learning models

# Importing specific classes from the swarmauri package
from swarmauri.standard.llms.concrete.GroqModel import GroqModel   # The model class to interact with GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage  # To define system messages
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent  # A basic conversation agent class
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation  # To handle conversation context

# Load environment variables from a .env file
load_dotenv()
# Fetch the API key stored in the .env file under the key "GROQ_API_KEY"
API_KEY = os.getenv("GROQ_API_KEY")

# Define the pre-defined system context
PREDEFINED_SYSTEM_CONTEXT = "You are a helpful assistant in driver training and first aid. Please provide accurate and concise information."

# Create an instance of the GroqModel using the API key
llm = GroqModel(api_key=API_KEY)

# Fetch the list of allowed models for the API key (these are models you can select from)
allowed_models = llm.allowed_models

# Initialize a conversation instance with maximum system context
conversation = MaxSystemContextConversation()

# Define a function to load a model by its name
def load_model(selected_model):
    # Create and return a new GroqModel instance with the selected model
    return GroqModel(api_key=API_KEY, name=selected_model)

# Define the main conversation function
# Takes user input, the conversation history, and model name as arguments
def converse(input_text, history, model_name):
    # Print the pre-defined system context and selected model for debugging
    print(f"system_context: {PREDEFINED_SYSTEM_CONTEXT}")
    print(f"Selected model: {model_name}")

    # Load the model based on the selected model name
    llm = load_model(model_name)
    
    # Create a conversation agent with the selected LLM and conversation instance
    agent = SimpleConversationAgent(llm=llm, conversation=conversation)
    
    # Set the system context for the conversation (pre-defined)
    agent.conversation.system_context = SystemMessage(content=PREDEFINED_SYSTEM_CONTEXT)
    
    # Convert input_text to a string (just to ensure it's properly formatted)
    input_text = str(input_text)
    
    # Print conversation and history (for debugging)
    print(conversation, history)
    
    # Execute the conversation agent's logic with the provided input
    result = agent.exec(input_text)
    
    # Print the result and its type (for debugging)
    print(result, type(result))
    
    # Return the result of the conversation as a string
    return str(result)

# Create a Gradio user interface (ChatInterface) for interacting with the system
demo = gr.ChatInterface(
    fn=converse,  # The function to handle conversations
    additional_inputs=[
        # Dropdown for selecting the model to be used in the conversation
        gr.Dropdown(label="Model Name", choices=allowed_models, value=allowed_models[0])        
    ],
    title="Driver Assistant",
    description="Ask for driving assistance"
)

if __name__ == "__main__":
    demo.launch()
