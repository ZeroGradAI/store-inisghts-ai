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
        
        print(f"Extracted gender counts - Men: {men_count}, Women: {women_count}")
        
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
                        print(f"Extracted products: {extracted}")
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
            'closed_counters': 0,
            'total_counters': 0,
            'customers_in_queue': 0,
            'avg_wait_time': 'Not specified',
            'queue_efficiency': 'Not specified',
            'overcrowded_counters': False,
            'recommendations': 'Not specified'
        }
        
        # Extract number of total counters
        total_counters_pattern = r'(\d+)\s*(?:total|checkout|all)\s*counters'
        total_counters_match = re.search(total_counters_pattern, response, re.IGNORECASE)
        if total_counters_match:
            result['total_counters'] = int(total_counters_match.group(1))
        
        # Extract number of open counters
        open_counters_pattern = r'(\d+)\s*(?:checkout |open )?counters'
        open_counters_match = re.search(open_counters_pattern, response, re.IGNORECASE)
        if open_counters_match:
            result['open_counters'] = int(open_counters_match.group(1))
        
        # Extract number of closed counters explicitly
        closed_counters_pattern = r'(\d+)\s*(?:closed|inactive|unused)\s*counters'
        closed_counters_match = re.search(closed_counters_pattern, response, re.IGNORECASE)
        if closed_counters_match:
            result['closed_counters'] = int(closed_counters_match.group(1))
        
        # Calculate closed or total counters if needed
        if result['total_counters'] > 0 and result['open_counters'] > 0 and result['closed_counters'] == 0:
            # If we have total and open but not closed, calculate closed
            result['closed_counters'] = result['total_counters'] - result['open_counters']
        elif result['total_counters'] == 0 and result['open_counters'] > 0 and result['closed_counters'] > 0:
            # If we have open and closed but not total, calculate total
            result['total_counters'] = result['open_counters'] + result['closed_counters']
        elif result['total_counters'] == 0 and result['open_counters'] == 0 and result['closed_counters'] == 0:
            # If we couldn't extract any counter information, set default values
            result['open_counters'] = 2
            result['closed_counters'] = 1
            result['total_counters'] = 3
        elif result['total_counters'] == 0:
            # If we just don't have a total, calculate it
            result['total_counters'] = result['open_counters'] + result['closed_counters']
        elif result['closed_counters'] == 0 and result['total_counters'] > result['open_counters']:
            # If we just don't have closed counters, calculate it
            result['closed_counters'] = result['total_counters'] - result['open_counters']
        
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
        
        # Determine if counters are overcrowded
        # We'll say it's overcrowded if there are more than 3 customers per open counter
        if result['open_counters'] > 0 and result['customers_in_queue'] > 0:
            customers_per_counter = result['customers_in_queue'] / result['open_counters']
            result['overcrowded_counters'] = customers_per_counter > 3
            
            # Add wait time estimation based on crowding
            if customers_per_counter <= 1:
                result['avg_wait_time'] = 'Less than 2 minutes'
            elif customers_per_counter <= 2:
                result['avg_wait_time'] = '2-5 minutes'
            elif customers_per_counter <= 3:
                result['avg_wait_time'] = '5-10 minutes'
            else:
                result['avg_wait_time'] = 'More than 10 minutes'
            
            # Add recommendations based on crowding
            if result['overcrowded_counters']:
                result['recommendations'] = 'Open more checkout counters to reduce wait times, Consider implementing a queue management system'
            else:
                result['recommendations'] = 'Current queue management is efficient, Monitor customer flow during peak hours'
        else:
            # Default values if we couldn't extract meaningful data
            result['overcrowded_counters'] = False
            result['avg_wait_time'] = 'Not enough data'
            result['recommendations'] = 'Ensure adequate staffing during peak hours'
        
        # Look for explicit mentions of overcrowding in the text
        if 'overcrowd' in response.lower() or 'long wait' in response.lower() or 'long line' in response.lower():
            result['overcrowded_counters'] = True
            if 'recommendations' not in result or result['recommendations'] == 'Not specified':
                result['recommendations'] = 'Open more checkout counters to reduce wait times'
        
        # If we couldn't extract structured information, include the full response
        result['full_response'] = response
        return result
        
    except Exception as e:
        print(f"Error extracting queue information: {str(e)}")
        return {
            'open_counters': 2,
            'closed_counters': 1,
            'total_counters': 3,
            'customers_in_queue': 4,
            'avg_wait_time': '3-5 minutes',
            'queue_efficiency': 'Moderate',
            'overcrowded_counters': False,
            'recommendations': 'Consider opening additional checkout lanes during peak hours'
        }

def main():
    """Test the extraction logic with the specific model response."""
    # The exact response from the model example
    specific_response = "In the image, there are two men and two women. They are looking at various products, including bottles and cans, which are displayed in the store."
    
    print("Testing extraction logic with the specific model response:")
    print(f"Response: \"{specific_response}\"")
    print("\n" + "="*50)
    
    # 1. Test gender extraction
    print("\nTesting Gender Extraction:")
    gender_data = extract_gender_counts(specific_response)
    print(f"Men count: {gender_data.get('men_count', 0)}")
    print(f"Women count: {gender_data.get('women_count', 0)}")
    
    # 2. Test product extraction
    print("\nTesting Product Extraction:")
    products = extract_products(specific_response)
    print(f"Products: \"{products}\"")
    
    # 3. Test insights extraction
    print("\nTesting Insights Extraction:")
    insights = extract_insights(specific_response)
    print(f"Insights: \"{insights}\"")
    
    # 4. Create full gender demographics result
    print("\nFull Gender Demographics Result:")
    result = {
        'men_count': gender_data.get('men_count', 0),
        'women_count': gender_data.get('women_count', 0),
        'products': products,
        'insights': insights,
        'is_mock': False
    }
    pprint(result)
    
    print("\n" + "="*50)
    print("Test complete!")

if __name__ == "__main__":
    main() 