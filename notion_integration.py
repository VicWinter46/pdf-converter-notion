import os
import json
import requests
import tempfile
import time
from datetime import datetime
import logging
from notion_client import Client
from dotenv import load_dotenv

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

# Notion API config
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# Initialize Notion client
notion = Client(auth=NOTION_API_KEY)

def get_pdfs_to_process():
    """Get PDFs with status 'Ready to Process' from Notion database"""
    try:
        response = notion.databases.query(
            database_id=DATABASE_ID,
            filter={
                "property": "Status",
                "select": {
                    "equals": "Ready to Process"
                }
            }
        )
        
        return response["results"]
    except Exception as e:
        logger.error(f"Error querying Notion database: {e}")
        return []

def download_pdf(url, filename):
    """Download a PDF from a URL"""
    try:
        response = requests.get(url)
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, filename)
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
            
        return pdf_path
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        return None

def update_page_status(page_id, status, csv_url=None, error=None):
    """Update the status of a page in Notion"""
    try:
        properties = {
            "Status": {
                "select": {
                    "name": status
                }
            }
        }
        
        if csv_url:
            properties["CSV URL"] = {
                "url": csv_url
            }
            
        if error:
            properties["Error"] = {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": error
                        }
                    }
                ]
            }
            
        notion.pages.update(
            page_id=page_id,
            properties=properties
        )
        
        logger.info(f"Updated page {page_id} status to {status}")
    except Exception as e:
        logger.error(f"Error updating Notion page: {e}")

def process_pdfs():
    """Main function to process PDFs from Notion"""
    pdfs = get_pdfs_to_process()
    logger.info(f"Found {len(pdfs)} PDFs to process")
    
    for page in pdfs:
        page_id = page["id"]
        logger.info(f"Processing page {page_id}")
        
        try:
            # Update status to Processing
            update_page_status(page_id, "Processing")
            
            # Find PDF file property
            pdf_property = None
            for prop_name, prop_data in page["properties"].items():
                if prop_data["type"] == "files" and prop_data["files"]:
                    pdf_property = prop_data
                    break
            
            if not pdf_property or not pdf_property["files"]:
                update_page_status(page_id, "Error", error="No PDF file found")
                continue
                
            # Get first PDF file
            pdf_file = pdf_property["files"][0]
            if pdf_file["type"] != "file":
                update_page_status(page_id, "Error", error="Not a file")
                continue
                
            pdf_url = pdf_file["file"]["url"]
            pdf_name = pdf_file["name"]
            
            # Download PDF
            pdf_path = download_pdf(pdf_url, pdf_name)
            if not pdf_path:
                update_page_status(page_id, "Error", error="Failed to download PDF")
                continue
                
            # Convert PDF to CSV
            config = load_config()
            csv_path = pdf_to_shopify_csv(pdf_path, config=config)
            
            # TODO: Upload CSV to a cloud storage service and get URL
            # For now, we'll just note that it was processed successfully
            
            # Update status to Completed
            update_page_status(
                page_id, 
                "Completed", 
                csv_url="https://example.com/csv-file", # Replace with actual URL
                error=None
            )
            
            # Clean up
            os.remove(pdf_path)
            
        except Exception as e:
            logger.error(f"Error processing page {page_id}: {e}")
            update_page_status(page_id, "Error", error=str(e))

if __name__ == "__main__":
    # Check if we have the required environment variables
    if not NOTION_API_KEY or not DATABASE_ID:
        logger.error("Missing required environment variables NOTION_API_KEY or NOTION_DATABASE_ID")
    else:
        # Run in a loop
        while True:
            process_pdfs()
            logger.info("Sleeping for 1 minute before checking again...")
            time.sleep(60)  # Check every minute
