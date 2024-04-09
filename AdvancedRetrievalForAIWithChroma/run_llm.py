from helper_utils import generate_content_for_query,augment_multiple_query,augment_query_generated,generate_data_for_ppt,generate_presentation_title,extract_json,formate_data_for_ppt
from ppt_gen import create_title_page,create_toc_slides
import json  # Import the json module

# content=generate_content_for_query("Presentation of summary Financial Report 2023","mistral","FormyconAG_EN_H1")
# # print("=====>",content)
# questions=augment_multiple_query("Presentation of summary Financial Report 2023",content,"mistral","FormyconAG_EN_H1",1)
# print(f"\n\n {questions} \n\n ")

print("\n\n Answer Generation started \n\n ")
response=[]
questions=[" What insights can be gained from comparing Formycon AG's asset and liability composition at the two reporting dates mentioned in the financial statements of their Half-Year Report 2023?"]
content="The query is asking for an analysis or insights that can be gained by comparing the composition of Formycon AG's assets and liabilities at the two reporting dates mentioned in their Half-Year Report 2023."
for question in questions:
    answer,content=augment_query_generated(question,content,"","dolphin2.1-mistral","FormyconAG_EN_H1")
    response.append({"question":question,"answer":answer,"content":content})
    
# title_str=generate_presentation_title("Presentation of summary Financial Report 2023","mistral","FormyconAG_EN_H1")
# print("\n\nTitle : =>",title_str)
# title = json.loads(title_str)

# ppt = create_title_page(title["title"],"TK AI")
# slide_data = ["Topic 1", "Topic 2", "Topic 3",  "Topic 3", "Topic 3","Topic 1", "Topic 2", "Topic 3",  "Topic 3", "Topic 3","Topic 1", "Topic 2", "Topic 3",  "Topic 3", "Topic 3","Topic 1", "Topic 2", "Topic 3",  "Topic 3", "Topic 3"]  # Add your topics here
# create_toc_slides(ppt, slide_data)
# ppt.save(f"{title["title"].replace(" ","-")}.pptx")

# for item in response:
#     slide_property_str=generate_data_for_ppt(item['question'],item['content'],item['answer'],"mistral","FormyconAG_EN_H1")
#     print("\n\n slide_property_str : ",slide_property_str)
#     slide_property_str=extract_json(slide_property_str)
#     print("\n\n extract_json : ",slide_property_str)
#     if slide_property_str!=None:
#         item["slide_property"]=(slide_property_str)
#         print("\n\n\n response : ",response[0]["slide_property"]['numberOfSlides'])
#         # print("\nd\n\n formate_data_for_ppt : \n\n",formate_data_for_ppt(response[0]["slide_property"]['slideStyle'],response[0]["slide_property"]['numberOfSlides'],response[0]["answer"],"mistral","FormyconAG_EN_H1"))



