from helper_utils import generate_content_for_query,augment_multiple_query,augment_query_generated,generate_data_for_ppt,generate_presentation_title,extract_json,formate_data_for_ppt
from ppt_gen import create_title_page,create_toc_slides,create_presentation_from_text,add_thank_you
import json  # Import the json module
from pptx import Presentation

import re



def sanitize_name(name):
    """
    Sanitize the provided name by replacing all special characters and whitespace
    with underscores, while also ensuring no leading or trailing underscores.

    Args:
        name (str): The string to be sanitized.

    Returns:
        str: The sanitized string, safe for use as an identifier.
    """
    # Replace all non-word characters (including whitespace) with underscores
    sanitized = re.sub(r'\W+', '_', name)
    
    # Remove leading/trailing underscores that may have been added
    sanitized = sanitized.strip('_')
    
    return sanitized


# context=generate_content_for_query("Presentation of summary Financial Report 2023","mistral","FormyconAG_EN_H1")
# # print("=====>",context)
# questions=augment_multiple_query("Presentation of summary Financial Report 2023",context,"mistral","FormyconAG_EN_H1",1)
# print(f"\n\n {len(questions)} \n\n ")

print("\n\n Answer Generation started \n\n ")
response=[]
questions=[" What insights can be gained from comparing Formycon AG's asset and liability composition at the two reporting dates mentioned in the financial statements of their Half-Year Report 2023?"]
context="The query is asking for an analysis or insights that can be gained by comparing the composition of Formycon AG's assets and liabilities at the two reporting dates mentioned in their Half-Year Report 2023."
title_str=generate_presentation_title("Presentation of summary Financial Report 2023","mistral","FormyconAG_EN_H1")
slide_property_str=extract_json(title_str)

print("\n\nTitle : =>",title_str)
# title = json.loads(title_str)
ppt = create_title_page(slide_property_str["title"],"TK AI")
slide_data = []  # Add your topics here

for question in questions:
    answer,context=augment_query_generated(question,context,"","mistral","FormyconAG_EN_H1")
    # formate_data_for_ppt(answer,"mistral","FormyconAG_EN_H1")
    response.append({"question":question,"answer":answer,"context":context})
    ppt=create_presentation_from_text(ppt,slide_data,answer.strip())
    

# input_text ="""
# # Asset Comparison
# - Total Assets: Q1 2023: €X.XX million, Q2 2023: €Y.YY million
# - Current Assets: Q1 2023: €A.AA million (X% of total assets), Q2 2023: €B.BB million (X% of total assets)
# - Non-Current Assets: Q1 2023: €C.CC million (Y% of total assets), Q2 2023: €D.DD million (Y% of total assets)

# # Liability Comparison
# - Total Liabilities: Q1 2023: €E.EE million, Q2 2023: €F.FF million
# - Current Liabilities: Q1 2023: €G.GG million (N% of total liabilities), Q2 2023: €H.HH million (N% of total liabilities)
# - Non-Current Liabilities: Q1 2023: €I.II million (P% of total liabilities), Q2 2023: €J.JJ million (P% of total liabilities)

# To determine the specific values and percentages, you would need to refer to the financial statements provided by Formycon AG. Insights that can be gained from comparing the asset and liability composition at these two reporting dates include changes in cash position, investment activities, debt levels, and overall financial health. For example, an increase in current assets could indicate increased inventory or accounts receivable, while a decrease in non-current liabilities could suggest a reduction in long-term debt.Presentation created successfully with improved formatting."""


# ppt = create_title_page(title["title"],"TK AI")
# ppt = Presentation()
# ppt=create_presentation_from_text(ppt,[],input_text.strip())
    
# create_toc_slides(ppt, slide_data)
ppt=add_thank_you(ppt)
ppt.save(f"{sanitize_name(slide_property_str["title"])}.pptx")
# ppt.save(f"{"title"}.pptx")
# for item in response:
#     slide_property_str=generate_data_for_ppt(item['question'],item['context'],item['answer'],"mistral","FormyconAG_EN_H1")
#     print("\n\n slide_property_str : ",slide_property_str)
#     slide_property_str=extract_json(slide_property_str)
#     print("\n\n extract_json : ",slide_property_str)
#     if slide_property_str!=None:
#         item["slide_property"]=(slide_property_str)
#         print("\n\n\n response : ",response[0]["slide_property"]['numberOfSlides'])
#         # print("\nd\n\n formate_data_for_ppt : \n\n",formate_data_for_ppt(response[0]["slide_property"]['slideStyle'],response[0]["slide_property"]['numberOfSlides'],response[0]["answer"],"mistral","FormyconAG_EN_H1"))




# Example usage: