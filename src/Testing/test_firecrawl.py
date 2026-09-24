from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os

load_dotenv(override = True)

app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
query = f"SDLC automation strategy for Tester, handling automated testing of a web application"

# Retrieve top web context
search_results = app.search(query=query)

# FIX: Add 'description' to the fallback chain. 
# It will look for 'markdown' -> 'content' -> 'description' -> and default to 'No content found'
context = "\n\n".join([
    getattr(item, 'markdown', 
        getattr(item, 'content', 
            getattr(item, 'description', 'No content found')
        )
    ) or '' # The 'or' handles cases where the attribute exists but is None
    for item in search_results.web
])

with open("search_result.txt", "a", encoding="utf-8") as f:
    f.write(context)

# This prints the first search result with all its available attributes
print(search_results.web[0].model_dump_json(indent=4))