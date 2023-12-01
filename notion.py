import requests
import json

NOTION_API_BASE_URL = "https://api.notion.com/v1"

def get_headers(api_key: str):
    return {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": "2021-08-16",
        "Content-Type": "application/json"
    }

def create_page(api_key,cover_url,parent_id,properties):
    data = {
        "parent": { "database_id": parent_id },
        "cover": {
            "type": "external",
            "external": {
                "url": cover_url
            }
        },
        "properties": properties
    }
    response = requests.post(f"{NOTION_API_BASE_URL}/pages", headers=get_headers(api_key), data=json.dumps(data))
    return response.json()