import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

file_name = "kaufland_faqs.csv"
faq_data = []

# Check if FAQ file already exists
if os.path.exists(file_name):
    print(f"{file_name} already exists. Loading data...")
    # Load data from file
    df = pd.read_csv(file_name)

else:    
    print(f"{file_name} does not exist. Scraping data...")

    url = "https://filiale.kaufland.de/service/haeufige-fragen.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
    }
    
    # Retrieve page data
    response = requests.get(url, headers=headers)
    parsed_content = BeautifulSoup(response.content, 'html.parser')
    
    # Target the titles directly as the starting point
    for title_element in parsed_content.find_all(class_="m-accordion__title"):
        
        # Extract the Question
        question = title_element.get_text(separator=" ", strip=True)
        
        # Find the corresponding Answer body that immediately follows this title
        body = title_element.find_next("div", {"class": "m-accordion__body"})
        
        if body:
            # Dive into the rich text container for the clean answer text
            richtext_div = body.find("div", {"class": "o-richtext"})
            
            if richtext_div:
                answer = richtext_div.get_text(separator=" ", strip=True)
                
                faq_data.append({
                    "Question": question, 
                    "Answer": answer
                })

    print(f"Collected {len(faq_data)} Q&A pairs.")

    # Save the data to CSV
    df = pd.DataFrame(faq_data)
    df.to_csv(file_name, index=False, encoding='utf-8-sig') # encoding ensures German characters are preserved
    print(f"Data saved to {file_name}")

# Preview the scraped data
print(df.head())