import os
import time
import tempfile
import logging
import json
import pickle
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import io

# Import the PDF converter
from pdf_converter import pdf_to_shopify_csv, load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File to keep track of processed files
PROCESSED_FILES_RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_files.json")

# Google Drive folder IDs - can be overridden by environment variables
TO_CONVERT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_TO_CONVERT_FOLDER_ID", "your-to-convert-folder-id")
CONVERTED_FOLDER_ID = os.getenv("GOOGLE_DRIVE_CONVERTED_FOLDER_ID", "your-converted-folder-id")
PROCESSED_FOLDER_ID = os.getenv("GOOGLE_DRIVE_PROCESSED_FOLDER_ID", "your-processed-folder-id")

# If modifying these scopes, delete the token.pickle file.
SCOPES = ['https://www.googleapis.com/auth/drive']

def load_processed_files():
    """Load the record of already processed files"""
    if os.path.exists(PROCESSED_FILES_RECORD):
        try:
            with open(PROCESSED_FILES_RECORD, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading processed files record: {str(e)}")
            return {}
    return {}

def save_processed_file(file_id, file_name):
    """Save a record of a processed file"""
    processed = load_processed_files()
    processed[file_id] = {
        "name": file_name,
        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(PROCESSED_FILES_RECORD, 'w') as f:
            json.dump(processed, f, indent=2)
        logger.info(f"Marked file {file_name} as processed")
    except Exception as e:
        logger.error(f"Error saving processed file record: {str(e)}")

def get_drive_service():
    """Get an authorized Google Drive service."""
    # Debug info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    print(f"Looking for credentials.json at: {os.path.join(os.getcwd(), 'credentials.json')}")
    print(f"credentials.json exists: {os.path.exists('credentials.json')}")
    print(f"token.pickle exists: {os.path.exists('token.pickle')}")
    
    creds = None
    
    # The token.pickle file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for token.pickle at: {os.path.join(os.getcwd(), 'token.pickle')}")
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials don't exist or are invalid, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                logger.error("credentials.json file not found. Please create it first.")
                return None
                
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    try:
        service = build('drive', 'v3', credentials=creds)
        logger.info("Successfully authenticated with Google Drive")
        return service
    except Exception as e:
        logger.error(f"Error building Drive service: {str(e)}")
        return None

def list_files_in_folder(service, folder_id):
    """List PDF files in a Google Drive folder"""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="files(id, name)"
        ).execute()
        
        files = results.get('files', [])
        logger.info(f"Found {len(files)} PDF files in folder {folder_id}")
        return files
    except Exception as e:
        logger.error(f"Error listing files in folder {folder_id}: {str(e)}")
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
        
        logger.info(f"Downloaded {file_name} to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {str(e)}")
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
        
        file_id = file.get('id')
        logger.info(f"Uploaded {filename} to Google Drive with ID: {file_id}")
        return file_id
    except Exception as e:
        logger.error(f"Error uploading file {filename}: {str(e)}")
        return None

def move_file(service, file_id, new_parent_id):
    """Move a file to a different folder in Google Drive"""
    try:
        # First check if the destination folder exists
        try:
            service.files().get(fileId=new_parent_id).execute()
        except Exception as e:
            logger.error(f"Destination folder {new_parent_id} does not exist: {str(e)}")
            return False
            
        # Get the current parents
        file = service.files().get(fileId=file_id, fields='parents').execute()
        previous_parents = ",".join(file.get('parents', []))
        
        if not previous_parents:
            logger.error(f"Could not determine parent folder for file {file_id}")
            return False
        
        # Move the file to the new folder
        service.files().update(
            fileId=file_id,
            addParents=new_parent_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        
        logger.info(f"Successfully moved file {file_id} to folder {new_parent_id}")
        return True
    except Exception as e:
        logger.error(f"Error moving file {file_id}: {str(e)}")
        return False

def try_alternative_file_handling(service, file_id, file_name):
    """Try alternative methods to handle a file if moving fails"""
    try:
        # Try to copy the file to processed folder
        file_metadata = {
            'name': file_name,
            'parents': [PROCESSED_FOLDER_ID]
        }
        
        # Create a copy in the processed folder
        service.files().copy(fileId=file_id, body=file_metadata).execute()
        logger.info(f"Copied file {file_id} to processed folder")
        
        # Try to trash the original file
        service.files().update(fileId=file_id, body={'trashed': True}).execute()
        logger.info(f"Moved original file {file_id} to trash")
        
        return True
    except Exception as e:
        logger.error(f"Error in alternative file handling for {file_id}: {str(e)}")
        return False

def process_pdfs():
    """Main function to process PDFs from Google Drive"""
    # Get Drive service
    service = get_drive_service()
    if not service:
        logger.error("Failed to create Google Drive service")
        return
    
    # Load already processed files
    processed_files = load_processed_files()
    logger.info(f"Loaded {len(processed_files)} previously processed files")
    
    # Get files to process
    files = list_files_in_folder(service, TO_CONVERT_FOLDER_ID)
    logger.info(f"Found {len(files)} PDFs in the To-Convert folder")
    
    # Filter out already processed files
    files_to_process = [f for f in files if f['id'] not in processed_files]
    logger.info(f"Of these, {len(files_to_process)} are new and will be processed")
    
    # Process each file
    for file in files_to_process:
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
            upload_success = upload_file(service, csv_path, csv_filename, CONVERTED_FOLDER_ID)
            
            if not upload_success:
                logger.error(f"Failed to upload CSV for {file_name}")
                continue
            
            # Try to move the original PDF to processed folder
            move_success = move_file(service, file_id, PROCESSED_FOLDER_ID)
            
            if not move_success:
                logger.warning(f"Could not move file {file_id} directly, trying alternative method")
                alt_success = try_alternative_file_handling(service, file_id, file_name)
                
                if not alt_success:
                    logger.warning(f"All methods to move/handle file {file_id} failed. File will remain in original location.")
            
            # Mark as processed regardless of move success
            save_processed_file(file_id, file_name)
            
            # Clean up temporary files
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists(csv_path):
                os.remove(csv_path)
            
            logger.info(f"Successfully processed {file_name}")
        
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")

def cleanup_old_records(days=30):
    """Clean up old records from the processed files list"""
    processed = load_processed_files()
    
    if not processed:
        return
    
    # Get current time
    now = datetime.now()
    
    # Filter out old records
    to_keep = {}
    removed = 0
    
    for file_id, info in processed.items():
        try:
            processed_time = datetime.strptime(info["processed_at"], "%Y-%m-%d %H:%M:%S")
            days_old = (now - processed_time).days
            
            if days_old <= days:
                to_keep[file_id] = info
            else:
                removed += 1
        except Exception:
            # Keep records with invalid dates
            to_keep[file_id] = info
    
    # Save updated records
    if removed > 0:
        try:
            with open(PROCESSED_FILES_RECORD, 'w') as f:
                json.dump(to_keep, f, indent=2)
            logger.info(f"Cleaned up {removed} old records from processed files list")
        except Exception as e:
            logger.error(f"Error saving cleaned records: {str(e)}")

def create_credentials_helper():
    """Helper function to guide users through creating credentials.json"""
    print("\n========== GOOGLE DRIVE SETUP GUIDE ==========")
    print("\n1. Go to https://console.developers.google.com/")
    print("2. Create a new project (or select an existing one)")
    print("3. Click 'Enable APIs and Services'")
    print("4. Search for 'Google Drive API' and enable it")
    print("5. Go to 'Credentials' in the left sidebar")
    print("6. Click 'Create Credentials' and select 'OAuth client ID'")
    print("7. Select 'Desktop app' as the Application type")
    print("8. Name it 'PDF to CSV Converter' and click 'Create'")
    print("9. Download the JSON file")
    print("10. Rename it to 'credentials.json' and place it in this directory")
    print("\nAfter completing these steps, run this script again.")
    print("==============================================\n")

if __name__ == "__main__":
    print("PDF to Shopify CSV Converter - Google Drive Integration")
    
    # Check if we have credentials.json
    if not os.path.exists('credentials.json'):
        create_credentials_helper()
        exit(1)
    
    # Check if we have the required folder IDs
    if TO_CONVERT_FOLDER_ID == "your-to-convert-folder-id" and not os.getenv("GOOGLE_DRIVE_TO_CONVERT_FOLDER_ID"):
        print("\nMissing required folder IDs. Please set these environment variables or edit the script:")
        print("GOOGLE_DRIVE_TO_CONVERT_FOLDER_ID - The folder to watch for new PDFs")
        print("GOOGLE_DRIVE_CONVERTED_FOLDER_ID - The folder to store converted CSVs")
        print("GOOGLE_DRIVE_PROCESSED_FOLDER_ID - The folder to move processed PDFs to")
        print("\nYou can find a folder ID in the URL when you open it in Google Drive:")
        print("https://drive.google.com/drive/folders/YOUR_FOLDER_ID")
        exit(1)
    
    print("\nStarting PDF converter. Press Ctrl+C to stop.")
    print(f"To-Convert Folder ID: {TO_CONVERT_FOLDER_ID}")
    print(f"Converted Folder ID: {CONVERTED_FOLDER_ID}")
    print(f"Processed Folder ID: {PROCESSED_FOLDER_ID}")
    
    # Run in a loop
    try:
import time

print("Starting PDF-to-Shopify processor... Watching Google Drive.")

while True:
    try:
        process_pdfs()  # Process any new PDFs in the folder
        print(f"Checked for PDFs at {time.strftime('%H:%M:%S')}. Waiting 60 seconds...")
        time.sleep(60)  # Wait for 60 seconds before checking again

    except Exception as e:
        print(f"⚠️ Error detected: {e}")
        time.sleep(30)  # Wait 30 seconds before retrying to prevent crash loops
