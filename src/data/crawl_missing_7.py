import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
import sys
import re
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DATA_DIR

# The 7 missing individual assessments you identified
missing_assessment_names = [
    'Customer Service Phone Solution',
    'Entry Level Cashier Solution', 
    'Entry Level Customer Service (General) Solution',
    'Entry Level Hotel Front Desk Solution',
    'Entry Level Sales Solution',
    'Entry Level Technical Support Solution',
    'Sales & Service Phone Solution'
]

def extract_test_types_working(soup):
    text = soup.get_text()
    match = re.search(r'Test Type:\s*([ABCDEKPS\s]+)', text)
    if match:
        found_text = match.group(1).strip()
        type_letters = found_text.split()
        valid_letters = [letter for letter in type_letters if letter in ['A', 'B', 'C', 'D', 'E', 'K', 'P', 'S']]
        
        type_mapping = {
            'A': 'Ability & Aptitude', 'B': 'Biodata & Situational Judgement', 'C': 'Competencies',
            'D': 'Development & 360', 'E': 'Assessment Exercises', 'K': 'Knowledge & Skills',
            'P': 'Personality & Behavior', 'S': 'Simulations'
        }
        return [type_mapping[letter] for letter in valid_letters if letter in type_mapping]
    return ['Knowledge & Skills']

def extract_description_fixed(soup):
    desc_elem = soup.find(string=lambda text: text and text.strip() == 'Description')
    if desc_elem:
        current = desc_elem.parent
        while current:
            next_elem = current.find_next_sibling()
            if next_elem:
                text = next_elem.get_text(strip=True)
                if len(text) > 50:
                    return text[:500]
            current = current.parent
    
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.get_text(strip=True)
        if (len(text) > 100 and 
            any(keyword in text.lower() for keyword in 
                ['test measures', 'measures knowledge', 'multi-choice test', 'assessment', 
                 'covers the following', 'designed for', 'evaluates', 'simulation'])):
            return text[:500]
    
    return 'Assessment description not found'

def extract_duration_fixed(soup):
    text = soup.get_text()
    patterns = [
        r'approximate completion time.*?(\d+)',
        r'completion time.*?(\d+)\s*minute',
        r'assessment length.*?(\d+)',
        r'(\d+)\s*minute'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            duration = int(match.group(1))
            if 1 <= duration <= 300:
                return duration
    return 60

def extract_job_levels(soup):
    text = soup.get_text()
    job_levels_match = re.search(r'Job levels\s*([^\n]+)', text, re.IGNORECASE)
    if job_levels_match:
        levels_text = job_levels_match.group(1).strip()
        levels = [level.strip().rstrip(',') for level in levels_text.split(',') if level.strip()]
        return levels
    return []

def find_and_crawl_missing_assessments():
    print('🔍 Finding and crawling the 7 missing assessments...')
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Find URLs for the missing assessments
    base_url = 'https://www.shl.com/products/product-catalog/'
    found_urls = []
    
    for start in range(0, 384, 12):
        page_url = f'{base_url}?start={start}&type=1'
        print(f'Searching page {start//12 + 1}/32...')
        
        try:
            response = session.get(page_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                name = a_tag.get_text(strip=True)
                
                if '/view/' in href and name in missing_assessment_names:
                    full_url = f'https://www.shl.com{href}' if href.startswith('/') else href
                    found_urls.append((name, full_url))
                    print(f'✅ Found: {name}')
        
        except Exception as e:
            print(f'Error: {e}')
        
        time.sleep(2)
    
    print(f'\\nFound {len(found_urls)} of the 7 missing assessments')
    
    # Crawl the found assessments
    new_assessments = []
    
    for name, url in found_urls:
        print(f'\\n📄 Crawling: {name}')
        
        try:
            response = session.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            description = extract_description_fixed(soup)
            test_types = extract_test_types_working(soup)
            duration = extract_duration_fixed(soup)
            job_levels = extract_job_levels(soup)
            
            assessment = {
                'url': url,
                'name': name,
                'adaptive_support': 'No',
                'description': description,
                'duration': duration,
                'remote_support': 'Yes',
                'test_type': test_types,
                'job_levels': job_levels
            }
            
            new_assessments.append(assessment)
            print(f'✅ Extracted: {name} | {test_types} | {duration}min')
            
        except Exception as e:
            print(f'❌ Error crawling {name}: {e}')
        
        time.sleep(3)
    
    # Load existing data and add new assessments
    existing_file = RAW_DATA_DIR / 'shl_assessments_complete.json'
    with open(existing_file, 'r') as f:
        existing_data = json.load(f)
    
    # Combine
    complete_data = existing_data + new_assessments
    
    # Save updated file
    output_file = RAW_DATA_DIR / 'shl_assessments_377_complete.json'
    with open(output_file, 'w') as f:
        json.dump(complete_data, f, indent=2, separators=(',', ': '))
    
    print(f'\\n🎉 Complete! Total assessments: {len(complete_data)}')
    print(f'📁 Saved to: {output_file}')

if __name__ == '__main__':
    find_and_crawl_missing_assessments()
