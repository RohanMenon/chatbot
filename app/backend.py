
from fastapi import FastAPI
from model import ModelInterface
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class Input(BaseModel):
    """Input text for the model."""
input_text: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_interface = ModelInterface()

@app.post("/chat_messages/")
def chat_messages(text: Input) -> dict:
    """Get a response from the model."""
    agent_response = model_interface.get_message_response(input_text=text.input_text)
    print(agent_response)
    return {"agent": agent_response["response"]}
    
@app.get("/status/")
def status():
    """Check the status of the API."""
    return{"status": "OK"}