from helper_utils import generate_content_for_query,augment_multiple_query,augment_query_generated

content=generate_content_for_query("Presentation of summary Financial Report 2023","mistral","FormyconAG_EN_H1")
# print("=====>",content)
questions=augment_multiple_query("Financial  Report 2023",content,"mistral","FormyconAG_EN_H1",10)
# # print(questions)
response=[]
for question in questions:
    answer=augment_query_generated(question,content,"","mistral","FormyconAG_EN_H1")
    response.append({"question":question,"answer":answer})
    
for item in response:
    print("\n question : ", item['question'],
          "\n answer : ", item['answer'])