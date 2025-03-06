import sys
import os
import argparse
import re
from pprint import pprint

def extract_gender_counts(response):
    """Extract gender counts from model response."""
    try:
        # Default values
        men_count = 0
        women_count = 0
        
        # Dictionary to convert word numbers to integers
        word_to_number = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        # Look for numbers of men and women in the response with digits
        digit_men_pattern = r'(\d+)\s*men'
        digit_women_pattern = r'(\d+)\s*women'
        
        # Look for numbers of men and women as words
        word_men_pattern = r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*men'
        word_women_pattern = r'(one|two|three|four|five|six|seven|eight|nine|ten)\s*women'
        
        # Search for both patterns for men
        digit_men_match = re.search(digit_men_pattern, response, re.IGNORECASE)
        word_men_match = re.search(word_men_pattern, response, re.IGNORECASE)
        
        # Search for both patterns for women
        digit_women_match = re.search(digit_women_pattern, response, re.IGNORECASE)
        word_women_match = re.search(word_women_pattern, response, re.IGNORECASE)
        
        # Extract men count
        if digit_men_match:
            men_count = int(digit_men_match.group(1))
        elif word_men_match:
            men_word = word_men_match.group(1).lower()
            men_count = word_to_number.get(men_word, 0)
        
        # Extract women count
        if digit_women_match:
            women_count = int(digit_women_match.group(1))
        elif word_women_match:
            women_word = word_women_match.group(1).lower()
            women_count = word_to_number.get(women_word, 0)
        
        return {
            'men_count': men_count,
            'women_count': women_count
        }
    except Exception as e:
        print(f"Error extracting gender counts: {str(e)}")
        return None

def extract_products(response):
    """Extract product information from model response."""
    try:
        # Look for mentions of products with different patterns
        products_patterns = [
            r'(looking at|browsing|viewing|shopping for|examining|checking|exploring)\s+(.*?)(?:\.|$)',
            r'products,\s+including\s+(.*?)(?:\.|$)',
            r'products\s+such\s+as\s+(.*?)(?:\.|$)',
            r'items\s+like\s+(.*?)(?:\.|$)',
            r'products,\s+which\s+include\s+(.*?)(?:\.|$)',
            r'products\s+(?:include|are)\s+(.*?)(?:\.|$)'
        ]
        
        # Try all patterns
        for pattern in products_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                # If it's the first pattern, we need group 2, otherwise group 1
                group_idx = 2 if pattern == products_patterns[0] else 1
                if len(match.groups()) >= group_idx:
                    extracted = match.group(group_idx).strip()
                    if extracted:
                        return extracted
        
        # If none of the specific patterns match, try more general cases
        if "products" in response.lower():
            # Find sentences containing "products"
            sentences = response.split('.')
            for sentence in sentences:
                if "products" in sentence.lower():
                    # Remove any leading phrases before "products"
                    if "products including" in sentence.lower():
                        parts = sentence.lower().split("products including")
                        if len(parts) > 1:
                            return parts[1].strip()
                    if "products such as" in sentence.lower():
                        parts = sentence.lower().split("products such as")
                        if len(parts) > 1:
                            return parts[1].strip()
                    return sentence.strip()
        
        return "Various store products"
    except Exception as e:
        print(f"Error extracting products: {str(e)}")
        return "Various store products"

def extract_insights(response):
    """Extract additional insights from model response."""
    try:
        gender_data = extract_gender_counts(response)
        if gender_data:
            men = gender_data.get('men_count', 0)
            women = gender_data.get('women_count', 0)
            
            if men == 0 and women == 0:
                # If we couldn't extract counts but have a response, try to provide some basic insight
                if "men" in response.lower() and "women" in response.lower():
                    return "The image shows a mix of male and female customers shopping in the store."
                else:
                    return "The image shows customers shopping in the retail environment."
            
            if men > women:
                return f"More men ({men}) than women ({women}) in the store, suggesting a male-oriented shopping experience."
            elif women > men:
                return f"More women ({women}) than men ({men}) in the store, suggesting a female-oriented shopping experience."
            else:
                return f"Equal number of men ({men}) and women ({women}) in the store, suggesting a balanced shopping environment."
        
        # Fallback to generic insights if we couldn't extract gender data
        if "products" in response.lower():
            for sentence in response.split('.'):
                if "products" in sentence.lower():
                    return f"Customers are browsing: {sentence.strip()}"
        
        return "Customers are shopping in the retail environment."
    except Exception as e:
        print(f"Error extracting insights: {str(e)}")
        return "Customers are shopping in the retail environment."

def extract_queue_info(response):
    """Extract queue management information from model response."""
    try:
        result = {
            'open_counters': 0,
            'customers_in_queue': 0,
            'avg_wait_time': 'Not specified',
            'queue_efficiency': 'Not specified',
            'recommendations': 'Not specified'
        }
        
        # Extract number of open counters
        counters_pattern = r'(\d+)\s*(?:checkout |open )?counters'
        counters_match = re.search(counters_pattern, response, re.IGNORECASE)
        if counters_match:
            result['open_counters'] = int(counters_match.group(1))
        
        # Extract number of customers in queue
        queue_pattern = r'(\d+)\s*customers?\s*(?:in|waiting|queuing)'
        queue_match = re.search(queue_pattern, response, re.IGNORECASE)
        if queue_match:
            result['customers_in_queue'] = int(queue_match.group(1))
        
        # Extract queue efficiency
        efficiency_pattern = r'queue management is\s*(\w+)'
        efficiency_match = re.search(efficiency_pattern, response, re.IGNORECASE)
        if efficiency_match:
            result['queue_efficiency'] = efficiency_match.group(1)
        
        # If we found at least some info, return the result
        if result['open_counters'] > 0 or result['customers_in_queue'] > 0:
            return result
        
        # If we couldn't extract structured information, include the full response
        result['full_response'] = response
        return result
        
    except Exception as e:
        print(f"Error extracting queue information: {str(e)}")
        return None

def analyze_gender_demographics(text):
    """Analyze the gender demographics from text."""
    gender_data = extract_gender_counts(text)
    products = extract_products(text)
    insights = extract_insights(text)
    
    return {
        'men_count': gender_data.get('men_count', 0),
        'women_count': gender_data.get('women_count', 0),
        'products': products,
        'insights': insights
    }

def analyze_queue_management(text):
    """Analyze the queue management from text."""
    return extract_queue_info(text)

def test_gender_extraction(sample_text):
    """Test gender demographics extraction logic."""
    print("\n===== TESTING GENDER DEMOGRAPHICS EXTRACTION =====")
    print(f"Sample text: \"{sample_text}\"")
    
    results = analyze_gender_demographics(sample_text)
    
    print("\nEXTRACTED RESULTS:")
    print(f"Men count: {results.get('men_count', 'Not found')}")
    print(f"Women count: {results.get('women_count', 'Not found')}")
    print(f"Products: {results.get('products', 'Not found')}")
    print(f"Insights: {results.get('insights', 'Not found')}")
    
    return results

def test_queue_extraction(sample_text):
    """Test queue management extraction logic."""
    print("\n===== TESTING QUEUE MANAGEMENT EXTRACTION =====")
    print(f"Sample text: \"{sample_text}\"")
    
    results = analyze_queue_management(sample_text)
    
    print("\nEXTRACTED RESULTS:")
    pprint(results)
    
    return results

def main():
    """Main function for running the extraction tests."""
    parser = argparse.ArgumentParser(description='Test extraction logic from model responses')
    parser.add_argument('--type', type=str, default='gender', choices=['gender', 'queue', 'both'],
                        help='Type of extraction to test (gender, queue, or both)')
    parser.add_argument('--text', type=str,
                        help='Sample text to use for testing (if not provided, sample texts will be used)')
    parser.add_argument('--samples', action='store_true',
                        help='Run tests with multiple sample texts')
    
    args = parser.parse_args()
    
    # Sample texts for each analysis type
    gender_samples = [
        "In the image, I can see 2 men and 3 women looking at various grocery products.",
        "The image shows two men and one woman browsing through electronics.",
        "I can see three women and four men in the clothing section.",
        "The store has approximately five female and six male customers looking at different items.",
        "In the retail space, there are two women and two men examining products on the shelves."
    ]
    
    queue_samples = [
        "There are 3 checkout counters open with about 5 customers waiting in line. The queue management seems efficient.",
        "I can see 2 checkout counters with approximately 7 customers in the queue. Wait time appears to be around 5-10 minutes.",
        "The store has 4 open counters but only 2 customers in line, suggesting very efficient queue management.",
        "There's only 1 checkout counter open with 8 customers waiting, indicating poor queue management.",
        "The image shows 3 checkout counters with 6 customers distributed evenly among them."
    ]
    
    if args.samples:
        # Run tests with multiple sample texts
        if args.type in ['gender', 'both']:
            for i, sample in enumerate(gender_samples):
                print(f"\nGender Sample #{i+1}:")
                test_gender_extraction(sample)
        
        if args.type in ['queue', 'both']:
            for i, sample in enumerate(queue_samples):
                print(f"\nQueue Sample #{i+1}:")
                test_queue_extraction(sample)
    else:
        # Run a single test with provided or default text
        if args.type in ['gender', 'both']:
            gender_text = args.text if args.text else gender_samples[0]
            test_gender_extraction(gender_text)
        
        if args.type in ['queue', 'both']:
            queue_text = args.text if args.text else queue_samples[0]
            test_queue_extraction(queue_text)
    
    print("\n===== EXTRACTION TESTING COMPLETE =====")

if __name__ == "__main__":
    main() 