from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOllama(model="qwen3:8b")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools = tools, verbose=True)
chain=agent_executor

def main():
    result = chain.invoke(
        input = {
            "input":"search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        }
    )

    print(result)

if __name__ == "__main__":
    main()
