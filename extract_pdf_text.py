import pypdf

def extract_text_from_pdf(pdf_path, output_path):
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Extracted {len(text)} characters.")
    print("First 1000 characters:")
    print(text[:1000])

if __name__ == "__main__":
    pdf_path = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier/NPCYCF documentation.pdf"
    output_path = "/Users/pulkitchauhan/Desktop/IDEAS_TIH/intent_clasifier/pdf_content.txt"
    extract_text_from_pdf(pdf_path, output_path)
