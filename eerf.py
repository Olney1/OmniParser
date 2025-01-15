import os
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

class FormElementDetector:
    def __init__(self):
        # Get the project root directory
        self.project_root = Path(__file__).parent
        
        # Initialize YOLO model
        self.yolo_model = get_yolo_model(
            model_path=str(self.project_root / "weights/icon_detect/best.pt")
        )
        
        # Initialize caption model
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=str(self.project_root / "weights/icon_caption_florence")
        )
        
        # Create temp directory in project root
        self.temp_folder = self.project_root / "temp_images"
        self.temp_folder.mkdir(exist_ok=True)

    def convert_pdf_page_to_image(self, pdf_path, page_num):
        """Convert a specific PDF page to image"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2, 2)  # Increase resolution
        pix = page.get_pixmap(matrix=mat)
        image_path = self.temp_folder / f"page_{page_num + 1}.png"
        pix.save(str(image_path))
        return str(image_path)

    def analyze_document(self, pdf_path, pages=None):
        """
        Analyze document pages for all form elements using OmniParser
        """
        results = []
        
        # Default to analyzing all pages if none specified
        if pages is None:
            doc = fitz.open(pdf_path)
            pages = range(len(doc))
        
        for page_num in pages:
            # Convert page to image
            image_path = self.convert_pdf_page_to_image(pdf_path, page_num)
            
            # Run OCR detection
            ocr_bbox_result, _ = check_ocr_box(
                image_path,
                display_img=False,
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={
                    'paragraph': False,
                    'text_threshold': 0.9
                },
                use_paddleocr=False
            )
            text_list, ocr_bbox = ocr_bbox_result
            
            # Run OmniParser detection
            _, _, parsed_content_list = get_som_labeled_img(
                image_path,
                self.yolo_model,
                BOX_TRESHOLD=0.05,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=None,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text_list,
                use_local_semantics=True,
                iou_threshold=0.9,
                prompt="Describe this form element"
            )
            
            # Add page number to results
            for element in parsed_content_list:
                element['page'] = page_num + 1
                element['image_path'] = image_path
            
            results.extend(parsed_content_list)
        
        return results

    def save_visualization(self, results, output_folder="output"):
        """
        Save visualizations of detected form elements
        """
        # Create output folder
        output_path = self.project_root / output_folder
        output_path.mkdir(exist_ok=True)
        
        # Group results by page
        from collections import defaultdict
        page_elements = defaultdict(list)
        for element in results:
            page_elements[element['image_path']].append(element)
        
        # Create visualization for each page
        for image_path, elements in page_elements.items():
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            # Draw each element
            for element in elements:
                # Get coordinates
                bbox = element['bbox']
                x1, y1 = int(bbox[0] * width), int(bbox[1] * height)
                x2, y2 = int(bbox[2] * width), int(bbox[3] * height)
                
                # Draw box
                color = (0, 255, 0) if element['interactivity'] else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{element['type']}: {element['content'][:30]}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save the annotated image
            output_file = output_path / f"page_{elements[0]['page']}_elements.png"
            cv2.imwrite(str(output_file), img)

def main():
    # Initialize detector
    detector = FormElementDetector()
    
    # Process document
    pdf_path = os.path.join(os.path.dirname(__file__), "60598751PV161001_EERF.pdf")
    print(f"Processing document: {pdf_path}")
    
    # Analyze specific pages (0-based indexing)
    results = detector.analyze_document(pdf_path, pages=[6, 8, 9])
    
    # Save visualizations
    detector.save_visualization(results)
    
    # Print results
    for element in results:
        print(f"\nPage {element['page']}:")
        print(f"Type: {element['type']}")
        print(f"Content: {element['content']}")
        print(f"Interactive: {element['interactivity']}")
        print(f"Coordinates: {element['bbox']}")

if __name__ == "__main__":
    main()