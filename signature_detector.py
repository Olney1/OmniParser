import os
from pathlib import Path
import cv2
import numpy as np
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import fitz  # PyMuPDF
from PIL import Image

class SignatureDetector:
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
        
        # Create temp directories
        self.temp_folder = self.project_root / "temp_images"
        self.signatures_folder = self.project_root / "extracted_signatures"
        self.temp_folder.mkdir(exist_ok=True)
        self.signatures_folder.mkdir(exist_ok=True)

    def _convert_pdf_page_to_image(self, pdf_path, page_num):
        """Convert a specific PDF page to image"""
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2, 2)  # Increase resolution
        pix = page.get_pixmap(matrix=mat)
        image_path = self.temp_folder / f"page_{page_num + 1}.png"
        pix.save(str(image_path))
        return str(image_path)

    def _extract_signature_region(self, image_path, bbox, page_num, sig_index):
        """Extract signature region from image"""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Convert relative coordinates to absolute
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        
        # Extract region
        signature_region = img[y1:y2, x1:x2]
        
        # Save signature region
        output_path = self.signatures_folder / f"page_{page_num}_sig_{sig_index}.png"
        cv2.imwrite(str(output_path), signature_region)
        
        return str(output_path)

    def detect_signatures(self, pdf_path, pages=None):
        """
        Detect and extract signature fields from a PDF document
        
        Returns:
        List of dictionaries containing:
            - page_number: Page where signature field was found
            - bbox: Relative coordinates [x1, y1, x2, y2]
            - contains_signature: Whether field appears to contain a signature
            - confidence: Confidence score of detection
            - signature_image: Path to extracted signature image
            - field_type: Type of signature field (e.g., 'owner signature', 'witness signature')
        """
        signatures = []
        
        # Default to analyzing all pages if none specified
        if pages is None:
            doc = fitz.open(pdf_path)
            pages = range(len(doc))
        
        for page_num in pages:
            # Convert page to image
            image_path = self._convert_pdf_page_to_image(pdf_path, page_num)
            
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
            
            # Filter for signature fields
            sig_index = 0
            for element in parsed_content_list:
                if 'signature' in element['content'].lower():
                    # Extract signature region
                    signature_image = self._extract_signature_region(
                        image_path, 
                        element['bbox'], 
                        page_num + 1, 
                        sig_index
                    )
                    
                    # Analyze if field contains a signature
                    # This is a simple check - could be enhanced with ML
                    img = cv2.imread(signature_image, cv2.IMREAD_GRAYSCALE)
                    std_dev = np.std(img)
                    contains_signature = std_dev > 30  # Threshold for non-empty field
                    
                    signatures.append({
                        'page_number': page_num + 1,
                        'bbox': element['bbox'],
                        'contains_signature': contains_signature,
                        'confidence': float(element.get('score', 0.0)),
                        'signature_image': signature_image,
                        'field_type': element['content'],
                        'interactivity': element['interactivity']
                    })
                    sig_index += 1
        
        return signatures

    def visualize_signatures(self, pdf_path, signatures, output_folder="signature_visualization"):
        """
        Create visualization of detected signature fields
        """
        output_path = self.project_root / output_folder
        output_path.mkdir(exist_ok=True)
        
        # Group signatures by page
        page_signatures = {}
        for sig in signatures:
            if sig['page_number'] not in page_signatures:
                page_signatures[sig['page_number']] = []
            page_signatures[sig['page_number']].append(sig)
        
        # Create visualization for each page
        for page_num, sigs in page_signatures.items():
            # Convert page to image if not already done
            image_path = self._convert_pdf_page_to_image(pdf_path, page_num - 1)
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            # Draw each signature field
            for sig in sigs:
                bbox = sig['bbox']
                x1, y1 = int(bbox[0] * width), int(bbox[1] * height)
                x2, y2 = int(bbox[2] * width), int(bbox[3] * height)
                
                # Color based on whether signature is detected
                color = (0, 255, 0) if sig['contains_signature'] else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{sig['field_type']}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save the annotated image
            output_file = output_path / f"page_{page_num}_signatures.png"
            cv2.imwrite(str(output_file), img)

def main():
    # Initialize detector
    detector = SignatureDetector()
    
    # Process document
    pdf_path = os.path.join(os.path.dirname(__file__), "60598751PV161001_EERF.pdf")
    print(f"Processing document: {pdf_path}")
    
    # Detect signatures (pages 7, 9, 10 in 0-based indexing)
    signatures = detector.detect_signatures(pdf_path, pages=[6, 8, 9])
    
    # Create visualization
    detector.visualize_signatures(pdf_path, signatures)
    
    # Print results
    for sig in signatures:
        print(f"\nPage {sig['page_number']}:")
        print(f"Field Type: {sig['field_type']}")
        print(f"Contains Signature: {sig['contains_signature']}")
        print(f"Confidence: {sig['confidence']:.2f}")
        print(f"Signature Image: {sig['signature_image']}")
        print(f"Interactive: {sig['interactivity']}")

if __name__ == "__main__":
    main() 