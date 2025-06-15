
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk

##### AIMessage #####
# from langchain_core.messages import HumanMessage, SystemMessage
# Useful Attributes:
# - content: The content of the message.
# - usage_metadata: Metadata about the usage of the message, such as token counts.
# - response_metadata["logprobs"]

##### Message with Image URL #####
# import base64
# import httpx
# from langchain_core.messages import HumanMessage

# image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
# image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
# message = HumanMessage(
#     content=[
#         {"type": "text", "text": "describe the weather in this image"},
#         {
#             "type": "image_url",
#             "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
#         },
#     ]
# )

##### set schema #####
# from langchain_core.pydantic_v1 import BaseModel, Field
# class GetWeatherData(BaseModel):
#     '''get weather data from a weather API'''
#     temperature: float = Field(..., description="Temperature in degrees Celsius")
#     humidity: float = Field(..., description="Humidity percentage")
#     condition: str = Field(..., description="Weather condition, e.g., sunny, rainy")

class IntegratedLLM:
    def __init__(self, vendor: str, model_name: str, logprobs: bool = False, base_url: str = None):
        self.vendor = vendor
        self.model_name = model_name
        self.logprobs = logprobs
        self.base_url = base_url

        assert vendor in ["openai", "anthropic", "google", "azure", "aws", "google_genai"], "Unsupported vendor"
        if vendor == "openai":
            # openai.OpenAI.chat.completions.create(...)
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                max_completion_tokens=512,
                timeout=None,
                max_retries=2,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        elif vendor == "anthropic":
            # pip install -U langchain-anthropic
            from langchain_anthropic import ChatAnthropic
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=0,
                max_tokens=512,
                timeout=None,
                max_retries=2,
                api_key=self.api_key,
            )
        elif vendor == "google":
            # pip install -U langchain-google-vertexai
            from langchain_google_vertexai import ChatGoogle
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set.")
            self.llm = ChatGoogle(
                model=model_name,
                temperature=0,
                max_tokens=512,
                timeout=None,
                max_retries=2,
                api_key=self.api_key,
            )
        elif vendor == "azure":
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")
            self.llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME", "default"),
                api_version=os.getenv("AZURE_API_VERSION", "2023-05-15"),
                model=model_name,
                temperature=0,
                max_tokens=512,
                timeout=None,
                max_retries=2,
            )
        elif vendor == "aws":
            from langchain_aws import ChatBedrockConverse
            # pip install -U langchain-aws
            self.llm = ChatBedrockConverse(
                model=model_name,
                temperature=0,
                max_tokens=512,
                timeout=None,
                max_retries=2,
            )
        elif vendor == "google_genai":
            # pip install -qU "langchain[google-genai]"
            from langchain.chat_models import init_chat_model
            self.llm = init_chat_model(model_name, model_provider="google_genai")

        if self.logprobs:
            self.llm = self.llm.bind(logprobs=True)

    def call(self, messages: list) -> AIMessage:
        """
        Call the LLM with the provided messages.
        :param messages: List of tuples (role, content)
        :return: Response from the LLM
        """
        response = self.llm.invoke(messages)
        return response
    
    def call_stream(self, messages) -> AIMessageChunk:
        """
        Call the LLM with the provided messages and return a generator for streaming responses.
        :param messages: List of tuples (role, content)
        :return: Generator yielding response chunks
        """
        response = self.llm.stream(messages)
        for chunk in response:
            yield chunk

    async def call_async(self, messages) -> AIMessage:
        """
        Asynchronously call the LLM with the provided messages.
        :param messages: List of tuples (role, content)
        :return: Response from the LLM
        """
        response = await self.llm.ainvoke(messages)
        return response
    
    async def call_stream_async(self, messages) -> AIMessageChunk:
        """
        Asynchronously call the LLM with the provided messages and return a generator for streaming responses.
        :param messages: List of tuples (role, content)
        :return: Generator yielding response chunks
        """
        response = await self.llm.astream(messages)
        async for chunk in response:
            yield chunk

    def bind_tools(self, tools: list):
        """
        Bind tools to the LLM.
        :param tools: List of tools to bind
        """
        self.llm.bind_tools(tools)

    def set_json_mode(self):
        return self.llm.bind(response_format={"type": "json_object"})
    
    def set_structured_mode(self, structure: dict):
        return self.llm.with_structured_output(structure)

    def get_model_name(self) -> str:
        """
        Get the model name of the LLM.
        :return: Model name
        """
        return self.model_name
    
    def get_vendor(self) -> str:
        """
        Get the vendor of the LLM.
        :return: Vendor name
        """
        return self.vendor
    
class IntegratedEmb:
    def __init__(self, vendor: str, model_name: str, base_url: str = None):
        self.vendor = vendor
        self.model_name = model_name
        self.base_url = base_url

        assert vendor in ["openai", "anthropic", "google", "azure", "aws"], "Unsupported vendor"
        if vendor == "openai":
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model=model_name, base_url=base_url)
        elif vendor == "anthropic":
            from langchain_anthropic import AnthropicEmbeddings
            self.embeddings = AnthropicEmbeddings(model=model_name)
        elif vendor == "google":
            from langchain_google_vertexai import GoogleVertexAIEmbeddings
            self.embeddings = GoogleVertexAIEmbeddings(model=model_name)
        elif vendor == "azure":
            from langchain_azure import AzureOpenAIEmbeddings
            self.embeddings = AzureOpenAIEmbeddings(model=model_name, base_url=base_url)
        elif vendor == "aws":
            from langchain_aws import BedrockEmbeddings
            self.embeddings = BedrockEmbeddings(model=model_name)

    def embed(self, texts: list) -> list:
        """
        Embed the provided texts.
        :param texts: List of texts to embed
        :return: List of embeddings
        """
        return self.embeddings.embed_documents(texts)
    

if __name__ == "__main__":
    # Example usage
    # llm = IntegratedLLM(vendor="azure", model_name="gpt-4.1-mini")
    # llm = IntegratedLLM(vendor="openai", model_name="gpt-4o-mini")
    llm = IntegratedLLM(vendor="openai", model_name="google/gemini-2.5-flash-preview-05-20", base_url=os.getenv("OPENAI_BASE_URL"))
    messages = [
        ("system", "You are a helpful assistant."),
        ("user", "What is the capital of France?"),
    ]
    response = llm.call(messages)
    print(response)  # Should print the response from the LLM