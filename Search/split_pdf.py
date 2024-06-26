import PyPDF2

def extract_first_10_pages(input_pdf_path, output_pdf_path):
    # Open the source PDF
    with open(input_pdf_path, 'rb') as infile:
        reader = PyPDF2.PdfReader(infile)
        writer = PyPDF2.PdfWriter()

        # Number of pages in the source PDF
        num_pages = len(reader.pages)

        # Loop through the first 10 pages or the total number of pages if less than 10
        for i in range(min(10, num_pages)):
            page = reader.pages[i]
            writer.add_page(page)

        # Write the output PDF
        with open(output_pdf_path, 'wb') as outfile:
            writer.write(outfile)

# Example usage
input_pdf_path = './pdf/KW23Abstracts (2).pdf'
output_pdf_path = './pdf/simple-data.pdf'
extract_first_10_pages(input_pdf_path, output_pdf_path)
