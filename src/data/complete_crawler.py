import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
import sys
import re
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DATA_DIR

def should_skip_assessment(name, url):
    skip_patterns = [
        'solution', 'short form', 'job focused', 'job-focused', 
        'package', 'bundle', 'suite'
    ]
    
    name_lower = name.lower()
    url_lower = url.lower()
    
    return any(pattern in name_lower or pattern in url_lower 
              for pattern in skip_patterns)

def extract_test_types_working(soup):
    text = soup.get_text()
    
    match = re.search(r'Test Type:\s*([ABCDEKPS\s]+)', text)
    if match:
        found_text = match.group(1).strip()
        type_letters = found_text.split()
        valid_letters = [letter for letter in type_letters if letter in ['A', 'B', 'C', 'D', 'E', 'K', 'P', 'S']]
        
        type_mapping = {
            'A': 'Ability & Aptitude',
            'B': 'Biodata & Situational Judgement', 
            'C': 'Competencies',
            'D': 'Development & 360',
            'E': 'Assessment Exercises',
            'K': 'Knowledge & Skills',
            'P': 'Personality & Behavior',
            'S': 'Simulations'
        }
        
        return [type_mapping[letter] for letter in valid_letters if letter in type_mapping]
    
    return ['Knowledge & Skills']

def extract_job_levels(soup):
    # Look for "Job levels" section
    text = soup.get_text()
    
    # Find job levels pattern
    job_levels_match = re.search(r'Job levels\s*([^\n]+)', text, re.IGNORECASE)
    if job_levels_match:
        levels_text = job_levels_match.group(1).strip()
        # Clean up and split job levels
        levels = [level.strip().rstrip(',') for level in levels_text.split(',') if level.strip()]
        return levels
    
    return []

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
    
    all_text = soup.get_text()
    lines = [line.strip() for line in all_text.split('\\n') if line.strip()]
    
    for i, line in enumerate(lines):
        if line.lower() == 'description' and i + 1 < len(lines):
            next_line = lines[i + 1]
            if len(next_line) > 50:
                return next_line[:500]
    
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

def extract_individual_assessment(url, session):
    try:
        response = session.get(url, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title_elem = soup.find('h1')
        name = title_elem.get_text(strip=True) if title_elem else 'Unknown'
        
        if should_skip_assessment(name, url):
            return None
        
        description = extract_description_fixed(soup)
        test_types = extract_test_types_working(soup)
        duration = extract_duration_fixed(soup)
        job_levels = extract_job_levels(soup)
        
        adaptive_support = 'No'
        remote_support = 'Yes'
        
        # Return with job_levels for internal use, API format for output
        return {
            'url': url,
            'name': name,
            'adaptive_support': adaptive_support,
            'description': description,
            'duration': duration,
            'remote_support': remote_support,
            'test_type': test_types,
            'job_levels': job_levels  # For internal recommendation matching only
        }
        
    except Exception as e:
        print(f'Error processing {url}: {e}')
        return None

def crawl_complete_shl_assessments():
    print('🎯 Complete SHL Crawler - All 377 Assessments with Job Levels...')
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    print('📋 Collecting all assessment URLs from 32 pages...')
    all_urls = []
    base_url = 'https://www.shl.com/products/product-catalog/'
    
    # All 32 pages: 31 pages × 12 + 1 page × 5 = 377 assessments
    for start in range(0, 384, 12):  # 0, 12, 24, ... 372
        page_url = f'{base_url}?start={start}&type=1'
        page_num = start//12 + 1
        print(f'📄 Page {page_num}/32...')
        
        try:
            response = session.get(page_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if '/view/' in href and '/products/product-catalog/view/' in href:
                    full_url = f'https://www.shl.com{href}' if href.startswith('/') else href
                    links.append(full_url)
            
            if not links:
                print(f'No links found on page {page_num}')
                break
                
            all_urls.extend(links)
            print(f'   Found {len(links)} assessment links')
            time.sleep(2)
            
        except Exception as e:
            print(f'Page {page_num} error: {e}')
            continue
    
    unique_urls = list(set(all_urls))
    print(f'📊 Processing {len(unique_urls)} unique URLs (expecting ~377)...')
    
    individual_assessments = []
    
    for i, url in enumerate(unique_urls, 1):
        print(f'[{i}/{len(unique_urls)}] Processing...')
        
        assessment = extract_individual_assessment(url, session)
        if assessment:
            individual_assessments.append(assessment)
            name = assessment['name']
            types = assessment['test_type']
            duration = assessment['duration']
            job_levels = assessment['job_levels']
            print(f'✅ {name} | {types} | {duration}min | Levels: {job_levels}')
        
        time.sleep(3)
    
    # Save complete dataset
    output_file = RAW_DATA_DIR / 'shl_assessments_complete.json'
    with open(output_file, 'w') as f:
        json.dump(individual_assessments, f, indent=2, separators=(',', ': '))
    
    print(f'\\n🎉 Complete crawl finished! {len(individual_assessments)} individual assessments')
    print(f'📁 Saved to: {output_file}')
    
    # Statistics
    type_counts = {}
    for assessment in individual_assessments:
        for test_type in assessment['test_type']:
            type_counts[test_type] = type_counts.get(test_type, 0) + 1
    
    print('\\n📊 Test Type Distribution:')
    for test_type, count in sorted(type_counts.items()):
        print(f'  {test_type}: {count}')
    
    print(f'\\n💡 Job levels included for internal matching - remove before API output')

if __name__ == '__main__':
    crawl_complete_shl_assessments()
