text="""
{
  "title": "Impact of Conditional Purchase Price Payments on Formycon AG's Financial Statements",
  "slideStyle": "summary",
  "numberOfSlides": 3,
  "fontSize": "24px",
  "extra": "Slide 1: Overview of the impact of conditional purchase price payments to Athos Group companies on Formycon AG's financial statements\nSlide 2: Breakdown of the balance sheet and income statement impact\nSlide 3: Summary and key takeaways"
}
The presentation will consist of three slides. The first slide provides an overview of the topic and its significance, while the second slide breaks down the balance sheet and income statement impact with clear diagrams or charts where applicable. The third slide summarizes the key findings from the previous slides. The font size for legibility on 4:3 aspect ratio slides designed in pixels is set to 24px."""


import json
import re

def extract_json(text):
    # Regular expression pattern to match a JSON object
    # This is a simplified pattern and might need adjustments for complex cases
    pattern = r'\{(?:[^{}]|(?:\{.*?\}))*\}'
    
    # Search for JSON object in the text
    text=text.replace("\n"," ")
    match = re.search(pattern, text)
    
    if match:
        # Extract the matched JSON text
        json_text = match.group(0)
        
        try:
            # Parse the JSON text to a Python dictionary
            json_obj = json.loads(json_text)
            return json_obj
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
    else:
        print("No valid JSON object found in the text.")
    
    return None
print(extract_json(text))