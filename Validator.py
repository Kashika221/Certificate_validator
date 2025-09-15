import os
import json
import logging
import tempfile
import shutil
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import traceback

from pdf2image import convert_from_path
import easyocr
from PIL import Image
from docx2pdf import convert
import cv2
from ultralytics import YOLO
from pydantic import BaseModel, ValidationError, Field
from groq import Groq
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('certificate_verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Student_Data(BaseModel):
    """Pydantic model for student data validation"""
    student_name: str = Field(..., min_length=1, max_length=100)
    enrollment_number: str = Field(..., min_length=1, max_length=50)
    cgpa: float = Field(..., ge=0.0, le=10.0)

class CertificateVerificationError(Exception):
    """Custom exception for certificate verification errors"""
    pass

class CertificateVerifier:
    def __init__(self):
        """Initialize the certificate verifier with necessary configurations"""
        self.temp_dir = None
        self.groq_client = None
        self.mongo_client = None
        self.ocr_reader = None
        self.base_certificate_path = None
        
        # Configuration
        self.SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.1"))
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
        self.SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.pdf', '.docx', '.bmp', '.tiff', '.gif'}
        
        self._setup()
    
    def _setup(self):
        """Setup all necessary components"""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="cert_verification_")
            logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Initialize OCR reader
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("OCR reader initialized")
            
            # Initialize Groq client
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise CertificateVerificationError("GROQ_API_KEY not found in environment variables")
            self.groq_client = Groq(api_key=groq_api_key)
            
            # Initialize MongoDB client
            mongo_uri = os.getenv("MONGO_URI") or os.getenv("MONGO_API_KEY")
            if not mongo_uri:
                raise CertificateVerificationError("MONGO_URI not found in environment variables")
            
            self.mongo_client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test connection
            self.mongo_client.admin.command('ping')
            logger.info("MongoDB connection established")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            self.cleanup()
            raise CertificateVerificationError(f"Setup failed: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary files and connections"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            
            if self.mongo_client:
                self.mongo_client.close()
                logger.info("MongoDB connection closed")
                
        except Exception as e:
            logger.warning(f"Cleanup warning: {str(e)}")
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file existence, size, and format"""
        try:
            if not os.path.exists(file_path):
                raise CertificateVerificationError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size > self.MAX_FILE_SIZE:
                raise CertificateVerificationError(
                    f"File too large: {file_size} bytes (max: {self.MAX_FILE_SIZE} bytes)"
                )
            
            if file_size == 0:
                raise CertificateVerificationError("File is empty")
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                raise CertificateVerificationError(
                    f"Unsupported file format: {file_ext}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"File validation failed: {str(e)}")
            raise
    
    def pdf_to_png(self, pdf_path: str) -> str:
        """Convert PDF to PNG with error handling"""
        try:
            logger.info(f"Converting PDF to PNG: {pdf_path}")
            pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
            
            if not pages:
                raise CertificateVerificationError("PDF appears to be empty or corrupted")
            
            png_path = os.path.join(self.temp_dir, f"pdf_output_{os.getpid()}.png")
            index = 0
            while os.path.isfile(png_path):
                png_path = os.path.join(self.temp_dir, f"img_output_{os.getpid()}{index}.png")
                index += 1
            pages[0].save(png_path, "PNG")
            logger.info(f"PDF converted to PNG: {png_path}")
            return png_path
            
        except Exception as e:
            logger.error(f"PDF to PNG conversion failed: {str(e)}")
            raise CertificateVerificationError(f"Failed to convert PDF: {str(e)}")
    
    def image_to_png(self, image_path: str) -> str:
        """Convert various image formats to PNG with error handling"""
        try:
            logger.info(f"Converting image to PNG: {image_path}")
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')
                
                png_path = os.path.join(self.temp_dir, f"img_output_{os.getpid()}.png")
                index = 0
                while os.path.isfile(png_path):
                    png_path = os.path.join(self.temp_dir, f"img_output_{os.getpid()}{index}.png")
                    index += 1
                img.save(png_path, "PNG")
                logger.info(f"Image converted to PNG: {png_path}")
                return png_path
                
        except Exception as e:
            logger.error(f"Image to PNG conversion failed: {str(e)}")
            raise CertificateVerificationError(f"Failed to convert image: {str(e)}")
    
    def docx_to_png(self, docx_path: str) -> str:
        """Convert DOCX to PNG via PDF with error handling"""
        try:
            logger.info(f"Converting DOCX to PNG: {docx_path}")
            
            temp_pdf = os.path.join(self.temp_dir, f"temp_{os.getpid()}.pdf")
            convert(docx_path, temp_pdf)
            
            if not os.path.exists(temp_pdf):
                raise CertificateVerificationError("DOCX to PDF conversion failed")
            
            return self.pdf_to_png(temp_pdf)
            
        except Exception as e:
            logger.error(f"DOCX to PNG conversion failed: {str(e)}")
            raise CertificateVerificationError(f"Failed to convert DOCX: {str(e)}")
    
    def convert_to_png(self, file_path: str) -> str:
        """Convert file to PNG format based on file extension"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.png':
                return file_path
            elif file_ext in {'.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}:
                return self.image_to_png(file_path)
            elif file_ext == '.pdf':
                return self.pdf_to_png(file_path)
            elif file_ext == '.docx':
                return self.docx_to_png(file_path)
            else:
                raise CertificateVerificationError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"File conversion failed: {str(e)}")
            raise
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR with error handling"""
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            if not os.path.exists(image_path):
                raise CertificateVerificationError(f"Image file not found: {image_path}")

            img = cv2.imread(image_path)
            results = self.ocr_reader.readtext(img)
            
            if not results:
                logger.warning("No text detected in image")
                return ""
            
            extracted_text = ""
            for bbox, text, confidence in results:
                if confidence > 0.5:
                    logger.debug(f"Detected text: {text} (Confidence: {confidence:.2f})")
                    extracted_text += " " + text.strip()
            
            logger.info(f"Text extraction completed. Length: {len(extracted_text)} characters")
            return extracted_text.strip()
        
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise CertificateVerificationError(f"Failed to extract text: {str(e)}")
    
    def extract_student_data(self, certificate_text: str) -> Student_Data:
        """Extract student data using Groq API with error handling"""
        try:
            logger.info("Extracting student data using LLM")
            
            if not certificate_text.strip():
                raise CertificateVerificationError("No text found in certificate")
            
            system_prompt = (
                "Extract the name of student, enrollment number and CGPA from the given text. "
                "Return the data in JSON format. If any field is not found, use appropriate default values. "
                "For CGPA, ensure it's a valid number between 0 and 10. "
                f"The JSON object must use the schema: {json.dumps(Student_Data.model_json_schema(), indent=2)}"
            )
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract data from this text: {certificate_text}"}
                ],
                model="openai/gpt-oss-120b",  # More reliable model
                temperature=0,
                max_tokens=1000,
                stream=False,
                response_format={"type": "json_object"},
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise CertificateVerificationError("No response from LLM")
            
            json_content = response.choices[0].message.content
            logger.debug(f"LLM response: {json_content}")
            
            # Validate and parse the response
            student_data = Student_Data.model_validate_json(json_content)
            logger.info(f"Extracted data - Name: {student_data.student_name}, "
                       f"Enrollment: {student_data.enrollment_number}, CGPA: {student_data.cgpa}")
            
            return student_data
            
        except ValidationError as e:
            logger.error(f"Data validation error: {str(e)}")
            raise CertificateVerificationError(f"Invalid student data format: {str(e)}")
        except Exception as e:
            logger.error(f"Student data extraction failed: {str(e)}")
            raise CertificateVerificationError(f"Failed to extract student data: {str(e)}")
    
    def verify_in_database(self, student_data: Student_Data) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify student data against database with error handling"""
        try:
            logger.info(f"Verifying student data in database: {student_data.enrollment_number}")
            
            db_name = os.getenv("MONGO_DB_NAME", "test")
            collection_name = os.getenv("MONGO_COLLECTION_NAME", "students")
            
            db = self.mongo_client[db_name]
            collection = db[collection_name]
            
            # Build query with fuzzy matching for name
            query = {
                "enrollmentNo": student_data.enrollment_number,
                "name": {"$regex": f"^{student_data.student_name}$", "$options": "i"},
                "cgpa": student_data.cgpa
            }
            
            logger.debug(f"Database query: {query}")
            record = collection.find_one(query)
            
            if record:
                logger.info("Record found in database")
                # Convert ObjectId to string for JSON serialization
                if '_id' in record:
                    record['_id'] = str(record['_id'])
                return True, record
            else:
                logger.info("No matching record found in database")
                return False, None
                
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Database connection error: {str(e)}")
            raise CertificateVerificationError(f"Database connection failed: {str(e)}")
        except Exception as e:
            logger.error(f"Database verification failed: {str(e)}")
            raise CertificateVerificationError(f"Database verification failed: {str(e)}")
    
    def verify_certificate(self, 
                          certificate_path: str, 
                          base_certificate_path: str) -> Dict[str, Any]:
        """Main method to verify certificate authenticity and extract data"""
        result = {
            "success": False,
            "similarity_score": 0.0,
            "student_data": None,
            "database_record": None,
            "verified": False,
            "message": "",
            "errors": []
        }
        
        try:
            logger.info(f"Starting certificate verification: {certificate_path}")
            
            # Validate input files
            self.validate_file(certificate_path)
            self.validate_file(base_certificate_path)
            
            # Convert files to PNG
            cert_png = self.convert_to_png(certificate_path)
            base_png = self.convert_to_png(base_certificate_path)
            
            # Calculate similarity
            similarity_score = self.similarity(cert_png, base_png)
            print("IN Validator.PY   \n \n")
            print(cert_png, " ", base_png)
            result["similarity_score"] = similarity_score
            
            if similarity_score < self.SSIM_THRESHOLD:
                result["message"] = f"Certificate format doesn't match expected template. Similarity: {similarity_score:.4f}"
                logger.warning(result["message"])
                return result
            
            logger.info(f"Certificate format verified. Similarity: {similarity_score:.4f}")
            
            # Extract text and student data
            certificate_text = self.extract_text_from_image(cert_png)
            student_data = self.extract_student_data(certificate_text)
            result["student_data"] = student_data.model_dump()
            
            # Verify against database
            verified, db_record = self.verify_in_database(student_data)
            result["verified"] = verified
            result["database_record"] = db_record
            
            if verified:
                result["success"] = True
                result["message"] = "Certificate verified successfully"
                logger.info("Certificate verification completed successfully")
            else:
                result["message"] = "Certificate data not found in database"
                logger.warning("Certificate data not verified in database")
            
            return result
            
        except CertificateVerificationError as e:
            logger.error(f"Certificate verification error: {str(e)}")
            result["errors"].append(str(e))
            result["message"] = str(e)
            return result
        except Exception as e:
            logger.error(f"Unexpected error during verification: {str(e)}")
            logger.error(traceback.format_exc())
            result["errors"].append(f"Unexpected error: {str(e)}")
            result["message"] = "An unexpected error occurred during verification"
            return result
        
    def detect_and_crop(self, image_path):
        """Detect certificate and crop the image, then display it"""
        model = YOLO("yolo_weights/best.pt")
        img = cv2.imread(image_path)

        results = model(img)[0]  # run detection

    # Get bounding box (x1, y1, x2, y2)
        box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        cropped = img[y1:y2, x1:x2]
        return cropped
    
    def similarity(self, img1, img2):
        try:
            logger.info(f"Calculating similarity between {img1} and {img2}")

            cropped_1 = self.detect_and_crop(img1)
            cropped_2 = self.detect_and_crop(img2)

            if cropped_1 is None:
                raise CertificateVerificationError(f"Cannot load image: {cropped_1}")
            if cropped_2 is None:
                raise CertificateVerificationError(f"Cannot load image: {cropped_2}")
            
            img1_gray = cv2.cvtColor(cropped_1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(cropped_2, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1_gray, None)
            kp2, des2 = sift.detectAndCompute(img2_gray, None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            score = len(good) / len(matches)
            logger.info(f"Similarity score: {score:.4f}")
            return score
        
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            raise CertificateVerificationError(f"Failed to calculate similarity: {str(e)}")


# CLI usage example
def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Certificate Verification System')
    parser.add_argument('certificate', help='Path to certificate file')
    parser.add_argument('base_certificate', help='Path to base certificate template')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)', default=None)
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    verifier = None
    try:
        verifier = CertificateVerifier()
        result = verifier.verify_certificate(args.certificate, args.base_certificate)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    finally:
        if verifier:
            verifier.cleanup()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())