import os
import time
import tempfile
import logging
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2 import service_account
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

# Google Drive folders
TO_CONVERT_FOLDER_ID = "1idz5ntPGvPpNbWtA6N_tckf_E8xy0ebm"
CONVERTED_FOLDER_ID = "1p03RvggjPXA0CUOD0gzpD0Ors7e9LQOF"
PROCESSED_FOLDER_ID = "1HkrKOWaPrIvz-6yJmgBtDm-VtW2MWKq_"

def get_drive_service():
    """Set up the Google Drive API service"""
    try:
        # Get credentials from environment
        credentials_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS")
        service_account_info = None
        
        if credentials_json:
            service_account_info = json.loads(credentials_json)
        else:
            # If not in environment, try to load from file
            credentials_file = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE", "credentials.json")
            if os.path.exists(credentials_file):
                with open(credentials_file, 'r') as f:
                    service_account_info = json.load(f)
        
        if not service_account_info:
            logger.error("No Google Drive credentials found")
            return None
            
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        logger.error(f"Error setting up Google Drive service: {e}")
        return None

def list_files_in_folder(service, folder_id):
    """List PDF files in a Google Drive folder"""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="files(id, name)"
        ).execute()
        
        return results.get('files', [])
    except Exception as e:
        logger.error(f"Error listing files in folder {folder_id}: {e}")
        return []

def download_file(service, file_id, file_name):
    """Download a file from Google Drive"""
    try:
        request = service.files().get_media(fileId=file_id)
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)
        
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                
        return file_path
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {e}")
        return None

def upload_file(service, file_path, filename, parent_folder_id):
    """Upload a file to Google Drive"""
    try:
        file_metadata = {
            'name': filename,
            'parents': [parent_folder_id]
        }
        
        media = MediaFileUpload(
            file_path,
            resumable=True
        )
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    except Exception as e:
        logger.error(f"Error uploading file {filename}: {e}")
        return None

def move_file(service, file_id, new_parent_id):
    """Move a file to a different folder in Google Drive"""
    try:
        # Get the current parents
        file = service.files().get(fileId=file_id, fields='parents').execute()
        previous_parents = ",".join(file.get('parents'))
        
        # Move the file to the new folder
        service.files().update(
            fileId=file_id,
            addParents=new_parent_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error moving file {file_id}: {e}")
        return False

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
