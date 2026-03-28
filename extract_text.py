import pypdf

def extract_text_from_pdf(pdf_path):
    text = ""
    
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        
        for page in reader.pages:
            text += page.extract_text()
    
    return text

extracted_text = extract_text_from_pdf("resume1.pdf")
print(extracted_text)