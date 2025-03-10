import os
import pickle
import time
import tempfile
import logging
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
import io

# Import the PDF converter
from pdf_converter import pdf_to_shopify_csv, load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Google Drive folders from environment variables
TO_CONVERT_FOLDER_ID = os.getenv('TO_CONVERT_FOLDER_ID')
CONVERTED_FOLDER_ID = os.getenv('CONVERTED_FOLDER_ID')
PROCESSED_FOLDER_ID = os.getenv('PROCESSED_FOLDER_ID')

def get_drive_service():
    """Authenticate and create Google Drive service using pickle token"""
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']

    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            logger.error(f"Error loading token.pickle: {e}")
            creds = None
    
    # If credentials are invalid or expired, try to refresh
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            logger.error(f"Error refreshing credentials: {e}")
            creds = None
    
    # If we don't have valid credentials, log an error
    if not creds or not creds.valid:
        logger.error("No valid credentials found. Please re-authenticate.")
        return None
    
    try:
        # Build and return the Drive service
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        logger.error(f"Error creating Drive service: {e}")
        return None

def process_pdfs():
    """Main function to process PDFs from Google Drive"""
    # Get Drive service
    service = get_drive_service()
    if not service:
        logger.error("Failed to create Google Drive service")
        return
    
    # Get files to process
    files = list_files_in_folder(service, TO_CONVERT_FOLDER_ID)
    logger.info(f"Found {len(files)} PDFs to process")
    
    # Process each file
    for file in files:
        file_id = file['id']
        file_name = file['name']
        logger.info(f"Processing file: {file_name}")
        
        try:
            # Download the PDF
            pdf_path = download_file(service, file_id, file_name)
            if not pdf_path:
                logger.error(f"Failed to download {file_name}")
                continue
            
            # Convert the PDF
            config = load_config()
            csv_path = pdf_to_shopify_csv(pdf_path, config=config)
            
            # Upload the CSV
            csv_filename = os.path.basename(csv_path)
            upload_file(service, csv_path, csv_filename, CONVERTED_FOLDER_ID)
            
            # Move the original PDF to processed folder
            move_file(service, file_id, PROCESSED_FOLDER_ID)
            
            # Clean up
            os.remove(pdf_path)
            
            logger.info(f"Successfully processed {file_name}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

# Keep the rest of the existing functions (list_files_in_folder, download_file, etc.)
# from your previous google_drive_integration.py script

if __name__ == "__main__":
    # Check if we have the required environment variables
    if not TO_CONVERT_FOLDER_ID or not CONVERTED_FOLDER_ID or not PROCESSED_FOLDER_ID:
        logger.error("Missing required folder IDs")
    else:
        # Run in a loop
        while True:
            process_pdfs()
            logger.info("Sleeping for 1 minute before checking again...")
            time.sleep(60)  # Check every minute
