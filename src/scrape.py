import requests
import re
from bs4 import BeautifulSoup
import pandas as pd

SURFACE_SPEED_URL = "https://www.tennisabstract.com/reports/atp_surface_speed.html"

# 2025 Season
def getTourneyUrls(session):
    response = session.get(SURFACE_SPEED_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find(id="reportable")  
    
    links = table.find_all("a")
    rows = table.find_all('tr')
    urls = []
    for row in rows:
        rowRes = []
        cells = row.find_all('td')
        for cell in cells:
            rowRes.append(cell)
        
        if not rowRes:
            continue   
        
        # Brisbane is part of the 2025 season but may be in 2024
        if "2024" in rowRes[0].text:
            if "Brisbane" not in rowRes[1].text:
                continue
        link = row.find('a')
        urls.append(link['href'])
    return urls

def parseTourneyResults(session, urls):
    count = 0
    for url in urls:
        rowsRes = []
        if count == 1:
            break
        count += 1
        response = session.get(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find(id="singles-results")

        # Finding header row
        header = table.find('thead')
        headerData = header.find_all('th')
        headerRow = []
        for cell in headerData:
            headerRow.append(cell.text)
        
        # Manually creating header labels
        headerRow[2] = 'Winner'
        headerRow[5] = 'Loser' 
        rowsRes.append(headerRow)

        # Finding table body and data
        body = table.find('tbody')
        rows = body.find_all('tr')
        for row in rows:
            rowRes = []
            cells = row.find_all('td')
            for cell in cells:    
                text = cell.text.strip()
                
                # Try to extract player names cleanly
                match = re.compile(r'^(?:\([^)]*\))?\s*([A-Za-z\s\'\-]+)\s*\[').search(text)
                if match:
                    text = match.group(1).strip()
                rowRes.append(text)
            rowsRes.append(rowRes)
        print(rowsRes[0])
        print('\n')
        print(rowsRes[1])
                
        


if __name__ == "__main__":
    session = requests.Session()
    urls = getTourneyUrls(session)
    parseTourneyResults(session, urls)



    
