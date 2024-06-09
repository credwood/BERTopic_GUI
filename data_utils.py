import os

from typing import List
import docx
from pdfminer.high_level import extract_text as pdf_extract_text
import textract

def extract_text_from_document(docs: List[str]):
    result = []
    for doc_path in docs:
        text = ""
        _, file_extension = os.path.splitext(doc_path)
        
        try:
            if file_extension.lower() == ".txt":
                with open(doc_path, "r", encoding="utf-8") as file:
                    text = file.read()
            elif file_extension.lower() == ".pdf":
                text = pdf_extract_text(doc_path)
            elif file_extension.lower() == ".docx":
                doc = docx.Document(doc_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
            else:
                # For other formats, try using textract
                try:
                    text = textract.process(doc_path, encoding='utf-8').decode('utf-8')
                except Exception as e:
                    print(f"Could not read {doc_path}: {e}")
                    continue
        
        except Exception as e:
            print(f"Could not read {doc_path}: {e}")
            continue

        result.append(text)
        
    return result

def get_document_paths(dir: str) -> List[str]:
    text_extensions = ['txt', 'doc', 'docx', 'pdf', 'rtf']
    text_files = [os.path.join(dir, f) for f in os.listdir(dir) 
                 if os.path.isfile(os.path.join(dir, f)) and  
                 f.split(".")[-1] in text_extensions
                 ]
    return text_files


    
