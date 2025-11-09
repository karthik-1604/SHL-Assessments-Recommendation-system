import requests
from bs4 import BeautifulSoup

def debug_test_types():
    url = 'https://www.shl.com/products/product-catalog/view/mongodb-new/'
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    response = session.get(url, timeout=25)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    print('🔍 All elements with product-catalogue__key class:')
    elements = soup.select('.product-catalogue__key')
    for i, elem in enumerate(elements):
        text = elem.get_text(strip=True)
        parent_text = elem.parent.get_text(strip=True)[:100] if elem.parent else 'No parent'
        print(f'{i+1}. Text: "{text}" | Parent context: "{parent_text}"')
    
    print(f'\\nTotal found: {len(elements)}')
    
    # Also check for other possible selectors
    print('\\n🔍 Looking for test type section:')
    test_section = soup.find(string=lambda text: text and 'test type' in text.lower())
    if test_section:
        print(f'Found test type text: {test_section}')
    else:
        print('No "test type" text found')

if __name__ == '__main__':
    debug_test_types()
