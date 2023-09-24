from decouple import config
import openai

# =======================================
OPENAI_API_KEY = config("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
# =======================================

# def get_completion(prompt, model="gpt-3.5-turbo"):

#     messages = [{
#         "role": "user",
#         "content": prompt
#     }]

#     """
#         According to https://platform.openai.com/docs/api-reference/chat/create

#         Given a prompt, the model will return one or more predicted completions, 
#         and can also return the probabilities of alternative tokens at each position

#         temperature:
#             Higher temperature values like 0.8 will make the output more random, while 
#             lower temperature values like 0.2 will make it more focused and deterministic.
#     """
#     response = openai.ChatCompletion.create(
#         model = model,
#         messages = messages,
#         temperature = 0.0
#     )


#     return response.choices[0].message["content"]

# =================================== #
# Usecase 1. 單純使用 OpenAI 做算術加減 #
# =================================== #

# ans = get_completion("What is 20-17?")

# ===================================================== #
# Usecase 2. 單純使用 OpenAI 把海盜似的文字轉譯成正式的英文 #
# ===================================================== #

# customer_email = "Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!"
# customer_style = "American English in a calm and respectful tone"
# prompt = f"Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{customer_email}```"
# print("========================")
# print(f"我對於AI的指令：\n{prompt}")
# print("========================\n")

# ans = get_completion(prompt)

# print("========================")
# print(f"AI的回覆：\n{ans}")
# print("========================\n")

# ======================================== #
# Usecase 3: 用 LangChain 感受下 modularity #
# ======================================== #
from langchain.chat_models import ChatOpenAI
"""
    According to https://api.python.langchain.com/en/latest/chat_models/langchain.chat_models.openai.ChatOpenAI.html

    We can set arguments like:
    (1) openai_api_key
    (2) temperature: default 0.7
    (3) model_name: default 'gpt-3.5-turbo' 
"""
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.0, model_name="gpt-3.5-turbo")

#from langchain.prompts import ChatPromptTemplate

# 注意這邊並不用特別的去加上 f-string
#template_string = "Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```"
#prompt_template = ChatPromptTemplate.from_template(template_string)
# 我能夠利用下面的 api 仔細查看每個細節
#prompt_template.messages[0].prompt.input_variables
#prompt_template.messages[0].prompt.template

# customer_messages = prompt_template.format_messages(
#     style=customer_style, # 這裡的 style 和 text 是我在 75 行設定的變數名稱
#     text=customer_email
# )

#customer_response = chat(customer_messages)
#print(customer_response.content)

# service_reply = "Hey there customer, the warranty does not cover cleaning expenses for your kitchen because it's your fault that you misused your blender by forgetting to put the lid on before starting the blender. Tough luck! See ya!"
# service_style_pirate = "a polite tone that speaks in English Pirate"
# service_messages = prompt_template.format_messages(
#     style=service_style_pirate,
#     text=service_reply
# )
# service_response = chat(service_messages)
# print(service_response.content)

# ====================================== #
# Usecase 4: 讓 LLM 的輸出結果具備格式型態 #
# ====================================== #

from langchain.prompts import ChatPromptTemplate

customer_review = """This leaf blower is pretty amazing.  It has four settings: candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's anniversary present. I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers out there, but I think it's worth it for the extra features."""

# # 下面這個 review template 會用來解析上面的文字，然後用 JSON 呈現
# review_template = """\
# For the following text, extract the following information:

# gift: Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.

# delivery_days: How many days did it take for the product to arrive? If this information is not found, output -1.

# price_value: Extract any sentences about the value or price, and output them as a comma separated Python list.

# Format the output as JSON with the following keys:
# gift
# delivery_days
# price_value

# text: {text}
# """

# prompt_template = ChatPromptTemplate.from_template(review_template)
# messages = prompt_template.format_messages(text=customer_review)
# response = chat(messages)

# # type of response.content is string
# print(response.content)

# 要把上面長得像是 JSON 的 string 真的變成 JSON
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift", description="Was the item purchased as a gift for someone else? Answer True if yes, False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days", description="How many days did it take for the product to arrive? If this information is not found, output -1.")
price_value_schema = ResponseSchema(name="price_value", description="Extract any sentences about the value or price, and output them as a comma separated Python list.")

response_schemas = [gift_schema, delivery_days_schema, price_value_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

# 這次這個 review_template_2 有使用到 format_instruction
review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt_template_2 = ChatPromptTemplate.from_template(template=review_template_2)
# 這個 format_instruction 使用上跟上面的 style 有點相似
messages = prompt_template_2.format_messages(text=customer_review, format_instructions=format_instructions)

response = chat(messages)

# 利用 output_parser 把 string 真的變成 json
output_dict = output_parser.parse(response.content)
print(output_dict["gift"])
print(output_dict["delivery_days"])
print(output_dict["price_value"])
