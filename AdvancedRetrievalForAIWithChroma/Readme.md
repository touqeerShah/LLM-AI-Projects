
```
python3 -m venv ./path/to/
source ./path/to/bin/activate
```

install 

pip3 install sentence-transformers

pip3 install numpy PyPDF2 tqdm
pip3 install langchain
pip3 install chromadb
pip3 install pypdf
pip3 install 'cryptography>=3.1'
pip install ollama
pip3 install -qU langchain pypdf llama-cpp-python huggingface_hub
pip3 install -qU sentence_transformers
pip3 install -q chromadb
pip3 install rank_bm25
pip3 install langchain-openai
pip3 install -q ragas
pip3 install gpt4all

python3 -m pip install python-pptx


question for slides
```
Contents: The current level of cash and cash equivalents as of June 30, 2023, for Formycon AG and its subsidiaries (Formycon Group) is €36,865K. This information can be found from the context provided in the "Interim Financial Statements" under the heading "Balance Sheet" where it shows a balance of cash and cash equivalents of €36,865,000 as of June 30, 2023
Based on contents give title for the slide, based on title what style summary , chart , or bulit points best to make slide, number of slides to display result, what will be front size  4:3(1280 x 720) so contents fit nicely , Final answer will be title ,slide style,number of slide and front size in json
```


code for calculate front size 
```
def calculate_font_size(text, max_length, min_font_size=12, max_font_size=24):
    """
    Calculates the font size based on the length of the text.

    Parameters:
    - text: The text string.
    - max_length: The maximum expected length of text for the max font size.
    - min_font_size: The minimum font size to use for long texts.
    - max_font_size: The maximum font size to use for short texts.

    Returns:
    - An integer representing the calculated font size.
    """
    text_length = len(text)
    
    if text_length <= max_length:
        return max_font_size
    else:
        # Calculate font size proportionally between min_font_size and max_font_size
        # This is a simple linear interpolation. Adjust the formula as needed.
        font_size = max_font_size - ((text_length - max_length) / max_length * (max_font_size - min_font_size))
        return max(min_font_size, int(font_size))

```
Step 2: Use the Function When Adding Text to a Slide

```
from pptx import Presentation
from pptx.util import Pt

# Initialize a presentation object
prs = Presentation()

# Add a slide
slide = prs.slides.add_slide(prs.slide_layouts[5])  # Change layout as needed

# Assume you have a textbox or placeholder
textbox = slide.shapes.add_textbox(left, top, width, height)  # Specify position and size
frame = textbox.text_frame
p = frame.add_paragraph()
text = "Your long or short text here"

# Calculate font size
max_text_length_for_max_font_size = 100  # This is an example value
font_size = calculate_font_size(text, max_text_length_for_max_font_size)

# Add text with calculated font size
p.text = text
p.font.size = Pt(font_size)

# Save the presentation
prs.save('adjusted_font_size_presentation.pptx')

```


Reference Link = [PDF Reference ](https://www.youtube.com/watch?v=ZN53DrEYwKg)