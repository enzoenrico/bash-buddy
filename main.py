import os
from typing import List
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_xai import ChatXAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from pydantic import SecretStr

# handling website loading & scraping
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer


from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys


load_dotenv()
api_key = SecretStr(os.getenv("XAI_API") or "")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
# llm = ChatXAI(model="grok-beta", temperature=0.8)

# prompting for the llm
template = """
You're an helpufull assistent chatbot, your user is a zoomer student, going to college, he's 19 and likes:
- to be on the internet
- coding
- artificial inteligence and machine learning
and based on what you learned about him, he also likes:
{interests}
remember to store his interests when possible, calling your available tools
anwser his questions like you're one of his friends, according to the context and message history below
\n
Message history: {messages} \n\n
Context: {context}\n\n
Respond to: {question}

"
"""

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="interests"),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def web_search(query: str) -> str:
    """
    Useful for searching current information from the internet. Use this when you need real-time data or facts.
    """
    search = DuckDuckGoSearchAPIWrapper(
        max_results=5,  # Limit number of results
        region="wt-wt",  # Worldwide results
        safesearch="off",  # Can be "strict", "moderate", or "off"
        time="d",  # Time period: 'd' for day, 'w' for week, 'm' for month
        backend="api",  # Use API backend for better results
    )
    res = search.run(query)

    docs = [Document(page_content=c) for c in res]
    splits = splitter.split_documents(docs)
    vector_store.add_documents(splits)

    return res[:500]


interests: List[str] = []


def end_chat(msg) -> bool:
    """
    Evaluate if the message is a quit command or a request to end the conversation
    """
    quit_commands = ["/exit", "quit", "bye", "goodbye", "end", "kill yourself"]
    if any(command in msg.lower() for command in quit_commands):
        sys.exit(1)
        return True
    return False


def remember(interaction_data: str):
    """
    Continuously analyze interactions to build and refine user profile and remember user provided data.
    Proactively identify and update:
    - Stated and implied interests
    - Conversation topics and engagement patterns
    - Mentioned activities and preferences
    - Behavioral traits and communication style
    - Time patterns and contextual information
    - Phrases, places, passwords and conversations

    Update profile when:
    - Direct mentions of likes/interests
    - Indirect references to activities and interests
    - Pattern changes in communication
    - New topics of enthusiasm
    - Recurring themes in conversations
    For example:
    "i really like kanye west"
    "i'm listening a lot to kanye recently"
    - The user deliberately asks you to remember something

    Store historical changes to track evolution of preferences.
    """

    interest_doc = Document(page_content=interaction_data)
    split_interest = splitter.split_documents([interest_doc])
    try:
        interests.append(interaction_data)
        vector_store.add_documents(split_interest)
        return True
    except:
        return False


# def access_link(links: List[str]):
#     """
#     Access a link passed by the user thorugh a chat conversation and get context based on it
#     Information of the website is stored in your vector store knowledge base
#     """

#     website_loader = AsyncChromiumLoader(links)
#     docs = website_loader.load()
#     print('loaded docs')
#     transformer = Html2TextTransformer()
#     print('transformed docs')

#     transformed_html = transformer.transform_documents(docs)

#     # default process for adding context
#     split_html = splitter.split_documents(transformed_html)
#     vector_store.add_documents(split_html)
#     print('added to vector store')
#     return transformed_html


tool_list = [
    Tool(
        name="web_search",
        func=web_search,
        description="Search the internet for the current information, if you're not sure about something or if the user requests questions on recent events, use this tool",
    ),
    Tool(
        name="end_chat",
        func=end_chat,
        description="Evaluate if the message is a quit command or a request to end the conversation",
    ),
    Tool(
        name="remember",
        func=remember,
        description="Based on what the user says, remember interests, hobbies, places, phrases, everything the user says, based on your judgement, store it",
    ),
    # Tool(
    #     name="access_link",
    #     func=access_link,
    #     description="Access and process content from a list of URLs provided in the conversation, extract relevant information and add it to the context",
    # ),
]

# vector db & loader
docs = [Document(page_content="FIRST DOCUMENT ")]

# search_res = DuckDuckGoSearchAPIWrapper().run("clima de hoje em curitiba - pr")

# for result in search_res:
#     docs.append(Document(page_content=result))

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

splits = splitter.split_documents(docs)

vector_store = FAISS.from_documents(
    splits,
    OpenAIEmbeddings(),
)

retriever = vector_store.as_retriever(kwargs={"k": 3})

# messages
# change the memory history
memory_buffer = ConversationBufferMemory(memory_key="messages", return_messages=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# agent creation
agent = create_openai_tools_agent(llm, tool_list, prompt=agent_prompt)
agent_executor = AgentExecutor(
    tools=tool_list, agent=agent, max_iterations=5, verbose=True
)

console = Console()


def display_welcome():
    welcome_msg = """
        - Type your questions and I'll help you out
        - Use /clear to clear screen
        - Use /exit to quit
        - Use /help for commands
    """
    console.print(
        Panel(welcome_msg, border_style="green", title="🤖 zpt - zoomer ai"),
    )


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_q():
    return input("write question here: \n [😼] > ")


def chat_loop():
    display_welcome()

    while True:
        try:
            # Get user input with styling
            question = console.input(f"\n[bold green]You[/bold green] > ")
            question = question.strip()

            # Handle commands
            if question.lower() == "/exit":
                console.print("[yellow]Goodbye![/yellow]")
                sys.exit(0)

            if question.lower() == "/clear":
                clear_screen()
                display_welcome()
                continue

            # Show searching animation
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Searching and thinking...[/bold blue]"),
                transient=True,
            ) as progress:
                progress.add_task("searching", total=None)

                # Get context and response
                context = retriever.invoke(question)
                context_text = format_docs(context)

                response = agent_executor.invoke(
                    {
                        "question": question,
                        "context": context_text,
                        "messages": memory_buffer.load_memory_variables({})["messages"],
                        "interests": interests,
                    }
                )

            # Display response in panel
            console.print(
                Panel(
                    response["output"],
                    title="[bold purple]Assistant[/bold purple]",
                    border_style="purple",
                ),
                Panel(
                    str(interests),
                    title="[bold red]Interests[/bold red]",
                    border_style="red",
                ),
            )

            # Update memory
            memory_buffer.save_context(
                {"input": question}, {"output": response["output"]}
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    chat_loop()
