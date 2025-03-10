import os
import logging
import time
import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from pdf_converter import pdf_to_shopify_csv, load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'path/to/service_account.json'

creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

service = build('drive', 'v3', credentials=creds)

def list_files(query):
    """List files in Google Drive matching the query."""
    try:
        results = service.files().list(
            q=query,
            pageSize=10,
            fields="nextPageToken, files(id, name)"
        ).execute()
        items = results.get('files', [])
        if not items:
            logger.info('No files found.')
        else:
            logger.info('Files:')
            for item in items:
                logger.info(f'{item["name"]} ({item["id"]})')
        return items
    except HttpError as error:
        logger.error(f'An error occurred: {error}')
        return None

def download_file(file_id, file_name):
    """Download a file from Google Drive."""
    try:
        request = service.files().get_media(fileId=file_id)
        with open(file_name, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                logger.info(f'Download {int(status.progress() * 100)}%.')
    except HttpError as error:
        logger.error(f'An error occurred: {error}')

def process_files():
    """Process files in Google Drive."""
    query = "mimeType='application/pdf'"
    files = list_files(query)
    if files:
        for file in files:
            file_id = file['id']
            file_name = file['name']
            local_path = f'/tmp/{file_name}'
            download_file(file_id, local_path)
            try:
                pdf_to_shopify_csv(local_path, config)
            except Exception as e:
                logger.error(f'Error converting {file_name}: {e}')

if __name__ == '__main__':
    while True:
        process_files()
        logger.info('Sleeping for 10 minutes...')
        time.sleep(600)
