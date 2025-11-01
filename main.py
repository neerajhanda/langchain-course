from dotenv import load_dotenv
from langchain_classic.chains.question_answering.map_rerank_prompt import output_parser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from os import environ

loaded = load_dotenv(encoding="utf-8-sig")
print(environ)
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm1 = ChatOllama(model="qwen3:8b")
llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
structured_llm = llm2.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input","agent_scratchpad","tool_names"]
).partial(format_instructions="")


agent = create_react_agent(
    llm = llm1,
    tools = tools,
    prompt=react_prompt_with_format_instructions
)

agent_executor = AgentExecutor(agent=agent, tools = tools, verbose=True)
chain=agent_executor
extract_output = RunnableLambda(lambda x:x["output"])
chain = agent_executor | extract_output | structured_llm

def main():
    result = chain.invoke(
        input = {
            "input":"search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        }
    )

    print(result)

if __name__ == "__main__":
    main()
