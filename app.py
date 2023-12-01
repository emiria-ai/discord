import discord
from discord import app_commands
from dotenv import load_dotenv
import openai
from openai import OpenAI
import json
import io
import re
import os
import requests
from typing import List
from uuid import uuid4
from urllib.parse import urlencode
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.utilities import SQLDatabase
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from tempfile import gettempdir
from transformers import pipeline
from notion import create_page
import chromadb

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///database.sqlite")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
GYAZO_IMAGE_UPLOAD_NDPOINT="https://upload.gyazo.com/api/upload"
GYAZO_ACCESS_TOKEN = os.environ.get("GYAZO_ACCESS_TOKEN") 

llm = LangChainOpenAI(temperature=0)

mbart_translator = pipeline('translation',
                            model='facebook/mbart-large-50-one-to-many-mmt',
                            src_lang='en_XX', tgt_lang='ja_XX')


openai_client = OpenAI()

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

Question: {input}"""

def gyazo_upload(image_url: str):
    headers = {'Authorization': "Bearer {}".format(GYAZO_ACCESS_TOKEN)}  

    response = requests.get(image_url)
    if response.status_code != 200:
        return None

    image_bytes = io.BytesIO(response.content)

    files = {'imagedata': image_bytes}  

    response = requests.post(GYAZO_IMAGE_UPLOAD_NDPOINT, headers=headers, files=files) 

    upload_data = response.json() 

    return upload_data["url"]

def create_dall_page(parent_id, cover_url,prompt, model,size):
    properties = {
        "prompt": {
            "title": [
                {
                    "text": {
                        "content": prompt
                    }
                }
            ]
        },
        "model": {
            "select": {
                "name": model
            }
        },
        "size": {
            "rich_text": [
                {
                    "text": {
                        "content": size
                    }
                }
            ]        
        }
    }
    
    return create_page(NOTION_API_KEY,cover_url,parent_id,properties)

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

db_chain = SQLDatabaseChain(llm=llm, database=db, prompt=PROMPT, verbose=True,return_sql=True)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

tree = app_commands.CommandTree(client)

categories = [
    {"name": "ニュース", "value": "news"},
    {"name": "スポーツ", "value": "sports"},
    {"name": "エンタメ", "value": "entertainment"}
]

def extract_file_extension(link_data):
    return link_data["file_path"].split('.')[-1]

def extract_github_link(message_obj):
    message_content = message_obj.content  # Extract the content from the Message object
    match = re.search(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/([^#]+)#L(\d+)C?\d?-L(\d+)", message_content)
    if match:
        return {
            "owner": match.group(1),
            "repo": match.group(2),
            "branch": match.group(3),
            "file_path": match.group(4),
            "start_line": int(match.group(5)),
            "end_line": int(match.group(6))
        }
    return None


async def fetch_code_from_github(link_data):
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw"
    }
    url = f"https://api.github.com/repos/{link_data['owner']}/{link_data['repo']}/contents/{link_data['file_path']}?ref={link_data['branch']}"
    response = requests.get(url, headers=headers)
    file_content = response.text
    lines = file_content.split('\n')
    return '\n'.join(lines[link_data['start_line']-1 : link_data['end_line']])

# 起動時に動作する処理
@client.event
async def on_ready():
    # 起動したらターミナルにログイン通知が表示される
    print('ログインしました')
    await tree.sync()

@client.event
async def on_message(message):

    if not message.author.bot:
        github_link_data = extract_github_link(message)
        if github_link_data:
            # GitHubのAPIを使用してコードを取得する処理を追加
            code = await fetch_code_from_github(github_link_data)
            extension = extract_file_extension(github_link_data)
            await message.reply(f"```{extension}\n{code}\n```")

    if message.author == client.user:
        return
    
    if client.user in message.mentions:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']):
                

                response = openai_client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": re.sub(r'<@!?(\d+)>', '', message.content).strip()},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": attachment.url,
                            },
                            },
                        ],
                        }
                    ],
                    max_tokens=300,
                )

                print(response.choices[0].message.content)

                await message.reply(response.choices[0].message.content)

    
def split_text_into_documents(long_string, max_docs=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
    )
    texts = text_splitter.split_text(long_string)
    docs = [Document(page_content=t) for t in texts[:max_docs]]

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

def add_documents(vectorstore: Chroma, documents: List[Document]):
    return vectorstore.add_documents(documents=documents)

persistent_client = chromadb.PersistentClient()

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="web",
    embedding_function=embedding_function,
)

@tree.command(name="qa",description="vector storeからQAを実行します")
async def qa(interaction: discord.Interaction,question: str):

    await interaction.response.defer()

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo"), chain_type="stuff",retriever=langchain_chroma.as_retriever(),return_source_documents=True)

    result = qa({'query': question})

    references = [doc.metadata for doc in result['source_documents']]

    print(references)
    await interaction.followup.send(result["result"])

@tree.command(name="add",description="add web vector data")
async def add(interaction: discord.Interaction,url: str):
                
    await interaction.response.defer()


    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    log = add_documents(langchain_chroma,all_splits)

    await interaction.followup.send("hi")

class SqlButton(discord.ui.View):
    def __init__(self, query):
        super().__init__()
        self.query = query
    
    @discord.ui.button(label='SQLを実行する', style=discord.ButtonStyle.blurple, row=4,emoji="▶️")
    async def pressedN(self, interaction: discord.Interaction, button: discord.ui.Button):
        # SQLクエリを実行
        query_result = db._execute(self.query)

        names = [entry['Name'] for entry in query_result]
        counts = [entry['TotalCount'] for entry in query_result]

        plt.figure(figsize=(10, 6))
        plt.barh(names, counts, color='skyblue')
        plt.xlabel('Count')
        plt.title('Name vs TotalCount')
        plt.gca().invert_yaxis()  # 上位の名前を上に表示
        plt.tight_layout()

        # io.BytesIOを使用して画像を一時的にメモリ上に保存
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()


        formatted_data = json.dumps(query_result, indent=4)
        message_to_send = f"```json\n{formatted_data}\n```"
        await interaction.response.send_message(f"クエリ結果: {message_to_send}",file=discord.File(buf, filename="result_plot.png"))


@tree.command(name="dalle",description="DALL-Eで画像を生成します")
@app_commands.choices(
    size=[
        app_commands.Choice(name='1024x1024', value='1024x1024'),
        app_commands.Choice(name='1792x1024', value='1792x1024'),
        app_commands.Choice(name='1024x1792', value='1024x1792'),
    ]
)
async def dall(interaction: discord.Interaction, prompt: str,size: app_commands.Choice[str]):

    await interaction.response.defer()

    model_name = "dall-e-3"

    try:
        response = openai_client.images.generate(
            model=model_name,
            prompt=prompt,
            size=size.value,
            quality="standard",
            n=1,
        )

        embed = discord.Embed(description=f"prompt: {prompt}",color=0x00bfff)
        image_url = gyazo_upload(response.data[0].url)
        embed.set_image(url=image_url)
        embed.set_author(name="OpenAI DALL-E 3",url="https://openai.com/dall-e-3",icon_url="https://pbs.twimg.com/profile_images/1634058036934500352/b4F1eVpJ_400x400.jpg")
        notion_result = create_dall_page(NOTION_DATABASE_ID,image_url,prompt,model_name,size.value)
        
        print(notion_result)
        await interaction.followup.send(embed=embed)

    except Exception as e:

        embed = discord.Embed(description=f"```{str(e)}```",color=0x00bfff)
        embed.set_author(name="OpenAI DALL-E 3",url="https://openai.com/dall-e-3",icon_url="https://pbs.twimg.com/profile_images/1634058036934500352/b4F1eVpJ_400x400.jpg")

        await interaction.followup.send(embed=embed)

@tree.command(name="carbon",description="carbon.now.shから美しいコード画像を生成します")
async def carbon(interaction: discord.Interaction, code: str, language: str):
    
    await interaction.response.defer()

    values = {
        "code": code,
        "language": language 
    }

    query_param = urlencode(values)

    baseURL = "https://carbon.vercel.app";
    url = f"{baseURL}?{query_param}"

    # Chrome のオプションを設定する
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')

    service = Service(ChromeDriverManager().install())
    browser = webdriver.Chrome(service=service, options=options)
    browser.set_window_size(1600, 1000)

    browser.get(url)

    container = browser.find_element(By.ID, "export-container")

    print(browser.current_url)

    download_path = os.path.join(gettempdir(), f"{uuid4()}.png")
    container.screenshot(download_path)

    await interaction.followup.send(file=discord.File(download_path, filename="carbon.png"))

@tree.command(name="hf_translate",description="facebook/mbart-large-50-one-to-many-mmt en -> ja")
async def hf_translate(interaction: discord.Interaction,text: str):
    
    await interaction.response.defer()

    hf_translate = mbart_translator(text)
    await interaction.followup.send(hf_translate[0]['translation_text'])


@tree.command(name="sql",description="自然言語からSQLを生成し、実行するかどうかを選択できます")
async def sql(interaction: discord.Interaction,input: str):
    
    await interaction.response.defer()

    result = db_chain(input)

    await interaction.followup.send(f'🔍 クエリー: {input}\n\n```{result["result"]}```', view=SqlButton(result["result"]))


client.run(os.getenv("DISCORD_BOT_TOKEN"))