import os
import requests
from bs4 import BeautifulSoup
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("website_scrapper")

@mcp.tool()
def website_scrapper(url: str) -> str:
    """Scrapes a specified website and extracts relevant information.
    Stores the extracted text content in a file named 'scraped_content.txt' in the same folder.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        str: The filename where the extracted information is saved.
    """
    # Use environment variable if needed (e.g., USER_AGENT) for polite scraping
    user_agent = os.getenv("USER_AGENT", "Mozilla/5.0 (compatible; MCPBot/1.0)")
    headers = {"User-Agent": user_agent}

    # Fetch website content
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise exception for HTTP errors

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract relevant text from website (for simplicity, all visible text)
    # We can join all text from body tag
    body = soup.body
    if body:
        text_content = body.get_text(separator="\n", strip=True)
    else:
        # Fallback: get all text
        text_content = soup.get_text(separator="\n", strip=True)

    # Store content in a file in the same folder
    filename = "scraped_content.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text_content)

    return filename

if __name__ == "__main__":
    mcp.run()
