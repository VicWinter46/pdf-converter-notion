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
    """Main function to process PDFs from Google Drive with enhanced logging"""
    # Log start of processing
    logger.info("Starting PDF processing...")
    
    # Log environment variables for debugging
    logger.info(f"Folder IDs - TO_CONVERT: {TO_CONVERT_FOLDER_ID}")
    logger.info(f"Folder IDs - CONVERTED: {CONVERTED_FOLDER_ID}")
    logger.info(f"Folder IDs - PROCESSED: {PROCESSED_FOLDER_ID}")
    
    # Get Drive service
    try:
        service = get_drive_service()
        if not service:
            logger.error("CRITICAL: Failed to create Google Drive service")
            return
    except Exception as e:
        logger.error(f"Unexpected error creating Drive service: {e}")
        return
    
    # Log files to process
    try:
        files = list_files_in_folder(service, TO_CONVERT_FOLDER_ID)
        logger.info(f"Found {len(files)} PDFs to process")
        
        # Log details of each file
        for file in files:
            file_id = file['id']
            file_name = file['name']
            logger.info(f"Processing file: {file_name} (ID: {file_id})")
            
            try:
                # Download the PDF
                pdf_path = download_file(service, file_id, file_name)
                if not pdf_path:
                    logger.error(f"FAILED to download {file_name}")
                    continue
                
                logger.info(f"Successfully downloaded {file_name} to {pdf_path}")
                
                # Convert the PDF
                config = load_config()
                csv_path = pdf_to_shopify_csv(pdf_path, config=config)
                
                logger.info(f"Successfully converted {file_name} to CSV at {csv_path}")
                
                # Upload the CSV
                csv_filename = os.path.basename(csv_path)
                csv_file_id = upload_file(service, csv_path, csv_filename, CONVERTED_FOLDER_ID)
                
                if csv_file_id:
                    logger.info(f"Successfully uploaded CSV: {csv_filename}")
                else:
                    logger.error(f"FAILED to upload CSV: {csv_filename}")
                
                # Move the original PDF to processed folder
                if move_file(service, file_id, PROCESSED_FOLDER_ID):
                    logger.info(f"Moved {file_name} to processed folder")
                else:
                    logger.error(f"FAILED to move {file_name} to processed folder")
                
                # Clean up
                os.remove(pdf_path)
                
                logger.info(f"Successfully processed {file_name}")
            
            except Exception as file_process_error:
                logger.error(f"CRITICAL ERROR processing {file_name}: {file_process_error}")
                # Optionally, you might want to log the full traceback
                import traceback
                logger.error(traceback.format_exc())
    
    except Exception as list_files_error:
        logger.error(f"Error listing files: {list_files_error}")

# Modify the main loop to include more robust error handling
if __name__ == "__main__":
    logger.info("Starting Google Drive PDF to CSV Conversion Service")
    
    # Check if we have the required environment variables
    if not TO_CONVERT_FOLDER_ID or not CONVERTED_FOLDER_ID or not PROCESSED_FOLDER_ID:
        logger.critical("MISSING REQUIRED FOLDER IDs. Cannot proceed.")
        # Exit the script if critical configuration is missing
        import sys
        sys.exit(1)
    
    # More robust continuous processing
    while True:
        try:
            process_pdfs()
            logger.info("Sleeping for 1 minute before next processing cycle...")
            time.sleep(60)  # Check every minute
        except Exception as main_loop_error:
            logger.critical(f"UNEXPECTED ERROR in main loop: {main_loop_error}")
            # Log the full traceback
            import traceback
            logger.critical(traceback.format_exc())
            
            # Wait a bit longer before retrying to prevent rapid error loops
            time.sleep(300)  # Wait 5 minutes before retrying
