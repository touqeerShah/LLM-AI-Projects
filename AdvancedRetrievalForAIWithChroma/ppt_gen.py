from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.text import MSO_AUTO_SIZE



def create_title_page(title, createdby):
    ppt = Presentation()
    # Setting Background
    slide_master = ppt.slide_master
    slide_master.background.fill.solid()
    slide_master.background.fill.fore_color.rgb = RGBColor(0, 0, 0)

    # Title Screen
    curr_slide = ppt.slides.add_slide(ppt.slide_layouts[0])
    curr_slide.shapes.title.text = title
    curr_slide.shapes.title.text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    curr_slide.shapes.title.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(
        255, 255, 255)
        # Check for a suitable placeholder for "Created By" text; if not found, create a new textbox
    if len(curr_slide.placeholders) > 1:  # Assumes placeholder 1 can be used
        created_by_placeholder = curr_slide.placeholders[1]
    else:
        # Define position and size for new textbox (adjust as necessary)
        left = Inches(1)
        top = Inches(5.5)
        width = Inches(6)
        height = Inches(1)
        created_by_placeholder = curr_slide.shapes.add_textbox(left, top, width, height)
    
    # Set "Created By" text
    created_by_placeholder.text = f"Created By: {createdby}"
    created_by_placeholder.text_frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    created_by_placeholder.text_frame.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    return ppt

def create_toc_slides(ppt, slide_data, title="Table of Contents"):
    max_entries_per_slide = 10  # Adjust based on your layout and font size
    slide_index = 0
    entry_index = 0

    for content in slide_data:
        # Create a new slide when starting or if the current slide is full
        if entry_index % max_entries_per_slide == 0:
            toc_slide = ppt.slides.add_slide(ppt.slide_layouts[1])
            title_placeholder = toc_slide.shapes.title
            title_placeholder.text = title if slide_index == 0 else f"{title} (cont.)"
             # Access the first paragraph in the text frame of the title placeholder
            # Assuming there's always at least one run in the first paragraph
            if title_placeholder.text_frame.paragraphs[0].runs:
                run = title_placeholder.text_frame.paragraphs[0].runs[0]
                run.font.color.rgb = RGBColor(255, 255, 255)
            else:  # If there are no runs, add one with the title text
                para = title_placeholder.text_frame.paragraphs[0]
                run = para.add_run()
                run.text = title if slide_index == 0 else f"{title} (cont.)"
                run.font.color.rgb = RGBColor(255, 255, 255)

            slide_index += 1
            entry_index = 0  # Reset for the new slide

        # Add content to the current slide
        tframe = toc_slide.placeholders[1].text_frame
        if entry_index == 0:  # Clear default text for the first entry
            tframe.clear()
        tframe.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        para = tframe.add_paragraph()
        para.text = content
        para.level = 1
        para.font.size = Pt(16)
        para.font.color.rgb = RGBColor(255, 255, 255)
        entry_index += 1

# Example usage
