import json

# Load the JSON data from the file
input_filename = 'scrapingdata.json'
with open(input_filename, 'r') as file:
    data = json.load(file)

# Iterate over each item in the list and remove the 'labels' key from 'paragraphsData'
for item in data:
    if "paragraphsData" in item and "labels" in item["paragraphsData"]:
        del item["paragraphsData"]["labels"]

# Write the modified data to a new JSON file
output_filename = 'scrapingdata.json'
with open(output_filename, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Modified data has been saved to {output_filename}.")
