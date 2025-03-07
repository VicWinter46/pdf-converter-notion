import os
import time
import tempfile
import logging
import json
import pickle

print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"Looking for credentials.json at: {os.path.join(os.getcwd(), 'credentials.json')}")
print(f"credentials.json exists: {os.path.exists('credentials.json')}")
print(f"token.pickle exists: {os.path.exists('token.pickle')}")
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for token.pickle at: {os.path.join(os.getcwd(), 'token.pickle')}")
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

# Google Drive folder IDs
TO_CONVERT_FOLDER_ID = "1idz5ntPGvPpNbWtA6N_tckf_E8xy0ebm"
CONVERTED_FOLDER_ID = "1p03RvggjPXA0CUOD0gzpD0Ors7e9LQOF"
PROCESSED_FOLDER_ID = "HkrKOWaPrIvz-6yJmgBtDm-VtW2MWKq_"

# If modifying these scopes, delete the token.pickle file.
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    """Get an authorized Google Drive service."""
    creds = None
    
    # The token.pickle file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
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
    
    return build('drive', 'v3', credentials=creds)

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
        
        logger.info(f"Uploaded {filename} to Google Drive with ID: {file.get('id')}")
        return file.get('id')
    except Exception as e:
        logger.error(f"Error uploading file {filename}: {e}")
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
            # Try to move the original PDF to processed folder
try:
    logger.info(f"Attempting to move file {file_id} to processed folder")
    move_success = move_file(service, file_id, PROCESSED_FOLDER_ID)
    
    if not move_success:
        logger.warning(f"Could not move file {file_id} to processed folder, trying to copy instead")
        # Alternative: Make a copy in the processed folder and delete original
        file_copy = service.files().get(fileId=file_id, fields='name').execute()
        file_metadata = {
            'name': file_copy.get('name'),
            'parents': [PROCESSED_FOLDER_ID]
        }
        
        # Create a copy in the processed folder
        service.files().copy(fileId=file_id, body=file_metadata).execute()
        
        # Delete the original file
        service.files().delete(fileId=file_id).execute()
        logger.info(f"Copied and deleted file {file_id} instead of moving")
except Exception as e:
    logger.error(f"Error handling file after processing: {str(e)}")
    logger.warning(f"File {file_id} was processed but remains in the original folder")
            
            # Clean up
            os.remove(pdf_path)
            
            logger.info(f"Successfully processed {file_name}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

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
        
    # Check if we have the required environment variables
    if not TO_CONVERT_FOLDER_ID or not CONVERTED_FOLDER_ID or not PROCESSED_FOLDER_ID:
        print("\nMissing required folder IDs. Please set these environment variables:")
        print("GOOGLE_DRIVE_TO_CONVERT_FOLDER_ID - The folder to watch for new PDFs")
        print("GOOGLE_DRIVE_CONVERTED_FOLDER_ID - The folder to store converted CSVs")
        print("GOOGLE_DRIVE_PROCESSED_FOLDER_ID - The folder to move processed PDFs to")
        print("\nYou can find a folder ID in the URL when you open it in Google Drive:")
        print("https://drive.google.com/drive/folders/YOUR_FOLDER_ID")
        exit(1)
        
    print("\nStarting PDF converter. Press Ctrl+C to stop.")
    
    # Run in a loop
    try:
        while True:
            process_pdfs()
            print(f"Checked for PDFs at {time.strftime('%H:%M:%S')}. Waiting 60 seconds...")
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\nStopping PDF converter.")