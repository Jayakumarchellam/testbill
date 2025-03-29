import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper
from datetime import datetime
import requests


app = Flask(__name__)
CORS(app)

llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    verbose=True
)

# Initialize tools
wikipedia = WikipediaAPIWrapper()

# Weather Tool
def get_weather(location):
    api_key = "a0b4c856135cbed33288672498d58a0e"  # Free tier available
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        weather_desc = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        return f"Weather in {location}: {weather_desc}, Temperature: {temp}°C, Humidity: {humidity}%"
    else:
        return f"Couldn't get weather for {location}. Error: {data.get('message', 'Unknown error')}"

# News Tool
def get_latest_news(topic="general"):
    api_key = "19ddde6b11c248baa548d778d963c5fa"  # Free tier available
    url = f"https://newsapi.org/v2/top-headlines?q={topic}&apiKey={api_key}"

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200 and data.get('articles'):
        articles = data['articles'][:3]
        result = f"Latest news about '{topic}':\n\n"
        for i, article in enumerate(articles, 1):
            result += f"{i}. {article['title']}\n"
            result += f"   Source: {article['source']['name']}\n"
            if article['description']:
                result += f"   {article['description']}\n"
            result += f"   Link: {article['url']}\n\n"
        return result
    else:
        return f"Couldn't find news about {topic}. Error: {data.get('message', 'Unknown error')}"

# Current date tool
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# Define custom tools
tools = [
    Tool(
        name="Weather",
        func=get_weather,
        description="Useful for getting weather information for a city or location"
    ),
    Tool(
        name="News",
        func=get_latest_news,
        description="Useful for getting latest news on a topic"
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for getting information from Wikipedia"
    ),
    Tool(
        name="Date",
        func=get_current_date,
        description="Useful for getting the current date"
    )
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
@app.before_request
def remove_ngrpk_warning():
    request.environ['ngrok-skip-browser-warning'] = 'true'


@app.route("/")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    chat_history = data.get("history", [])

    # Create context from history
    context = ""
    if chat_history:
        for msg in chat_history[-5:]:
            role = "Human" if msg.get("isUser") else "AI"
            context += f"{role}: {msg.get('text')}\n"

    try:
        # Format prompt for Llama 2
        formatted_prompt = f"<s>[INST] Previous conversation:\n{context}\n\nUser question: {user_message} [/INST]"

        # Run the agent with the user's message
        response = agent.run(formatted_prompt)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"response": f"I encountered an error processing your request. Please try again."})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
