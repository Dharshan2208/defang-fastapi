from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


app = FastAPI()
templates = Jinja2Templates(directory="templates")

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6,api_key=GOOGLE_API_KEY)
tool = TavilySearch(max_results=10, return_direct=True,api_key=TAVILY_API_KEY)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)
graph_builder.set_entry_point("chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
thread_id = str(uuid.uuid4())

class ChatMessage(BaseModel):
    role: str
    content: str


chat_history: List[ChatMessage] = []


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "chat_history": chat_history}
    )


@app.post("/send")
async def send_message(request: Request, message: str = Form(...)):
    global chat_history
    chat_history.append(ChatMessage(role="user", content=message))

    # Run through LangGraph
    config = {"configurable": {"thread_id": thread_id}}
    response_text = ""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": message}]}, config=config
    ):
        for value in event.values():
            response_text = value["messages"][-1].content

    chat_history.append(ChatMessage(role="assistant", content=response_text))
    return RedirectResponse(url="/", status_code=303)
