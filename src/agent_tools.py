#environment variables
from dotenv import load_dotenv
import os
# Imports for the retriever tool
from langchain.tools.retriever import create_retriever_tool
# imports to chat with the LLM and keep messages in the state
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
# Pydantic for class definition for model
from pydantic import BaseModel, Field
from typing import Literal
# imports to assemble the graph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
# import to display the graph
from IPython.display import Image, display


# Load environment variables from .env file
load_dotenv()
# get the directory where to persist the database
retriever_name = os.getenv("RETRIEVER_NAME", "retrieve_RAG")
retriever_description = os.getenv("RETRIEVER_DESCRIPTION", "Search and return information about EOG performance.")
chat_model = os.getenv("CHAT_MODEL", "openai:gpt-4.1")
grader = os.getenv("GRADER_MODEL", "openai:gpt-4.1")
# a simple query to test the retriever
simple_query = "what can you say about the culture?"

# the model we're using and temp=0 to not halucitante
response_model = init_chat_model(chat_model, temperature=0)

# this model can be different to the one used previusly, maybe a reasoning llm
grader_model = init_chat_model(grader, temperature=0)

# prompt to grade the documents retrieved
GRADE_PROMPT = os.getenv("GRADE_PROMPT")

# prompt to rewrite the question if we receive irrelevant documents
REWRITE_PROMPT = os.getenv("REWRITE_PROMPT")

# prompt if the question was ok from the user
GENERATE_PROMPT = os.getenv("GENERATE_PROMPT")

# graph image path
image_path = os.getenv("GRAPH_IMAGE_PATH", "./state_graph.png")


def get_retriever_tool(vectorstore, name, description):
    """
    Create a retriever tool from the given vectorstore.
    Args:
        vectorstore: The vectorstore to be used for creating the retriever.
        name (str): The name of the retriever tool.
        description (str): The description of the retriever tool.
    Returns:
        retriever_tool: The created retriever tool.
    """
    
    # Create the retriever tool with a retriever from vectorstore, name, description
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        name,
        description,
    )
    return retriever_tool

def simple_query(retriever_tool, query) -> list:
    """
    Perform a simple query using the given retriever.
    Args:
        retriever_tool: The retriever to be used for the query.
        query (str): The query string.
    Returns:
        list: A list of documents retrieved by the query.
    """
    # try a query
    return retriever_tool.invoke({"query": query})

# function that decides if query the RAG or direct response from LLM
def generate_query_or_respond(state: MessagesState, retriever_tool) -> dict:
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])  
    )
    return {"messages": [response]}


# class used to create objects from the documents and give them a grade
class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

# function to grade each document receives the state and decides if answer or
# rewrite the question
def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(  
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
    
# function that rewrites the original question to get better responses
def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

# function to generate answer if the question was ok from the user
def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

def generate_graph(state: MessagesState, retriever_tool) -> StateGraph:
    """Generate the state graph for the agent."""
    # initiate the workflow
    workflow = StateGraph(state)

    # Define the nodes we will cycle between
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile
    graph = workflow.compile()

    # save the graph as an image
    Image(filename=image_path)
    
    #display the graph as an image
    display(Image(graph.get_graph().draw_mermaid_png()))

    return graph