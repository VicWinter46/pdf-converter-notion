import pdfplumber
import openai
import pandas as pd
import os
import json
import logging
import sys
from io import StringIO
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_converter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration management
def load_config():
    """Load configuration from config.json file or create default if not exists"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
    if not os.path.exists(config_path):
        default_config = {
            "openai_api_key": "",
            "model": "gpt-3.5-turbo",
            "output_dir": "output",
            "watch_dir": "watch",
            "prompt_template": """
Extract product details in CSV format:
Product Title,Body (HTML),Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color

If MSRP is missing, use Wholesale Price.
Sizes should be standardized (XS, S, M, L, XL, 0-3 M, 2T, etc.).
Only include Color if multiple colors exist.

Text:
{text}

Return only CSV data.
"""
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        logger.warning(f"Created default config file at {config_path}. Please edit it with your API key.")
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for required fields
        required_fields = ["openai_api_key", "model", "output_dir", "watch_dir", "prompt_template"]
        for field in required_fields:
            if field not in config or not config[field]:
                if field == "openai_api_key":
                    logger.error(f"Missing {field} in config file. Please add it to {config_path}")
                    raise ValueError(f"Missing {field} in config file")
        
        # Create directories if they don't exist
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["watch_dir"], exist_ok=True)
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

def process_with_openai(text, config):
    """Process text with OpenAI API - compatible with older OpenAI library"""
    try:
        # Set the API key for the older OpenAI library version
        openai.api_key = config["openai_api_key"]
        
        # Use an enhanced prompt to ensure we get cost data and product type
        enhanced_prompt = """
Extract product details in CSV format with these EXACT columns:
Product Title,Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color

IMPORTANT INSTRUCTIONS:
- Product Type is CRUCIAL and must be extracted correctly (e.g. Apparel, Footwear, Accessories, etc.)
- Wholesale Price MUST be included - this is the cost per item
- If MSRP is missing, use Wholesale Price * 2
- Sizes should be standardized (XS, S, M, L, XL, 0-3 M, 2T, etc.)
- Only include Color if multiple colors exist
- Ensure every product has a SKU
- Use double quotes around any field containing commas
- Ensure proper CSV formatting with exactly 8 columns per row

Text from purchase order:
{text}

Return ONLY the CSV data without any explanations or markdown formatting.
"""
        
        # Use the enhanced prompt instead of the template from config
        response = openai.ChatCompletion.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a specialized assistant that precisely extracts structured product data from purchase orders for Shopify import."},
                {"role": "user", "content": enhanced_prompt.format(text=text)}
            ],
            temperature=0.1  # Lower temperature for more consistent results
        )
        
        return response['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error processing with OpenAI: {str(e)}")
        raise
def parse_csv_data(csv_data):
    """Parse CSV data into a DataFrame with enhanced error handling"""
    try:
        # Clean up the data to handle potential formatting issues
        csv_lines = csv_data.strip().split('\n')
        
        # Extract header
        if not csv_lines:
            raise ValueError("Empty CSV data")
        
        header = csv_lines[0]
        expected_columns = len(header.split(','))
        
        # Fix lines with too many columns
        fixed_lines = [header]
        for i in range(1, len(csv_lines)):
            line = csv_lines[i].strip()
            if not line:
                continue
                
            # Try to fix problematic lines
            fields = line.split(',')
            if len(fields) > expected_columns:
                # This is a line with too many commas
                # Try to identify fields that might contain commas
                fixed_line = []
                field_buffer = []
                count = 0
                
                for field in fields:
                    field_buffer.append(field)
                    count += 1
                    
                    # If we've reached the expected number of columns or this is the last field
                    if count == expected_columns or field == fields[-1]:
                        if count < expected_columns:
                            # We don't have enough fields yet, continue collecting
                            continue
                        
                        # Add the collected fields to the fixed line
                        fixed_line.extend(field_buffer[:-1])
                        
                        # Combine any remaining fields
                        if field_buffer:
                            fixed_line.append(" ".join(field_buffer[-1:]))
                            
                        break
                    
                fixed_lines.append(",".join(fixed_line))
            else:
                fixed_lines.append(line)
        
        clean_csv = '\n'.join(fixed_lines)
        
        # Log the cleaned CSV for debugging
        logger.info("Cleaned CSV data:")
        logger.info(clean_csv)
        
        # First try with standard parsing
        try:
            products_df = pd.read_csv(StringIO(clean_csv))
            return products_df
        except Exception as first_error:
            # If standard parsing fails, try with more flexible options
            try:
                logger.info("Standard CSV parsing failed, trying with quoting options...")
                products_df = pd.read_csv(StringIO(clean_csv), quoting=1)  # QUOTE_ALL
                return products_df
            except Exception:
                try:
                    logger.info("Trying with error_bad_lines=False...")
                    # For older pandas versions
                    products_df = pd.read_csv(StringIO(clean_csv), error_bad_lines=False)
                    return products_df
                except Exception:
                    try:
                        logger.info("Trying with on_bad_lines='skip'...")
                        # For newer pandas versions
                        products_df = pd.read_csv(StringIO(clean_csv), on_bad_lines='skip')
                        return products_df
                    except Exception:
                        # One last attempt: manual parsing
                        try:
                            logger.info("Trying manual parsing...")
                            # Split by lines and then by commas
                            header_fields = fixed_lines[0].split(',')
                            data = []
                            
                            for i in range(1, len(fixed_lines)):
                                row = {}
                                fields = fixed_lines[i].split(',')
                                
                                # Match available fields to header
                                for j in range(min(len(header_fields), len(fields))):
                                    row[header_fields[j].strip()] = fields[j].strip()
                                    
                                data.append(row)
                                
                            # Create DataFrame
                            products_df = pd.DataFrame(data)
                            return products_df
                        except Exception as e:
                            # If all attempts fail, log the original error
                            logger.error(f"All CSV parsing attempts failed: {str(first_error)}")
                            raise first_error
    except Exception as e:
        logger.error(f"Error parsing CSV data: {str(e)}")
        
        # Print the raw CSV data for debugging
        logger.error("Raw CSV data for debugging:")
        logger.error(csv_data)
        
        raise

def format_for_shopify(products_df):
    """Format the DataFrame for Shopify import"""
    try:
                    # Ensure required columns exist
        required_columns = ["MSRP", "Wholesale Price", "SKU", "Size", "Color", "Product Type"]
        for col in required_columns:
            if col not in products_df.columns:
                products_df[col] = None
        
        # Calculate price (use MSRP if available, otherwise use Wholesale Price * 2)
        def calculate_price(row):
            if "MSRP" in row and pd.notna(row["MSRP"]) and row["MSRP"] != 0:
                return row["MSRP"]
            elif "Wholesale Price" in row and pd.notna(row["Wholesale Price"]):
                return row["Wholesale Price"] * 2  # Default markup
            else:
                return None
        
        # Map columns to Shopify format
        shopify_mapping = {
            "Product Title": "Title",
            "SKU": "SKU",
            "Vendor": "Vendor",
            "Product Type": "Type",
            "Wholesale Price": "Cost per item",
            "MSRP": "Price",
            "Size": "Option1 value",
            "Color": "Option2 value"
        }
        
        # Create new DataFrame with Shopify columns
        shopify_df = pd.DataFrame()
        
        # Map existing columns to Shopify format
        for old_col, new_col in shopify_mapping.items():
            if old_col in products_df.columns:
                shopify_df[new_col] = products_df[old_col]
        
        # Set default values for required fields
        shopify_df["URL handle"] = shopify_df["Title"].str.lower().str.replace(' ', '-').str.replace('[^a-z0-9-]', '', regex=True)
        shopify_df["Description"] = shopify_df["Title"]  # As requested, just use title for description
        shopify_df["Status"] = "draft"  # Set status to draft
        shopify_df["Option1 name"] = "Size"  # Default option name
        shopify_df["Option2 name"] = "Color"  # Default option name
        shopify_df["Continue selling when out of stock"] = "FALSE"
        shopify_df["Product image URL"] = ""  # Empty by default
        
        # Make sure we have a price
        if "Price" not in shopify_df.columns and "Cost per item" in shopify_df.columns:
            # If no MSRP was found, use Cost per item as Price
            shopify_df["Price"] = shopify_df["Cost per item"]
        
        # Select only the columns requested by the user
        shopify_columns = [
            "Title", "URL handle", "Description", "Vendor", "Type", 
            "Status", "SKU", "Option1 name", "Option1 value", 
            "Option2 name", "Option2 value", "Price", "Cost per item",
            "Continue selling when out of stock", "Product image URL"
        ]
        
        # Filter only columns that exist in the DataFrame
        existing_columns = [col for col in shopify_columns if col in shopify_df.columns]
        
        # For any missing columns in the requested list, add them with empty values
        for col in shopify_columns:
            if col not in shopify_df.columns:
                shopify_df[col] = ""
        
        shopify_csv = shopify_df[shopify_columns]
        
        # Log what's being returned for debugging
        logger.info(f"Returning Shopify CSV with columns: {', '.join(shopify_csv.columns)}")
        
        return shopify_csv
    
    except Exception as e:
        logger.error(f"Error formatting for Shopify: {str(e)}")
        raise

def pdf_to_shopify_csv(pdf_path, output_path=None, config=None):
    """Convert a PDF to a Shopify-compatible CSV"""
    if config is None:
        config = load_config()
    
    if output_path is None:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config["output_dir"], f"{pdf_name}_{timestamp}.csv")
    
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Process with OpenAI
        csv_data = process_with_openai(text, config)
        
        # Parse CSV data
        products_df = parse_csv_data(csv_data)
        
        # Format for Shopify
        shopify_csv = format_for_shopify(products_df)
        
        # Save to CSV
        shopify_csv.to_csv(output_path, index=False)
        
        logger.info(f"Successfully converted {pdf_path} to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting PDF to CSV: {str(e)}")
        raise

def process_directory(directory=None, config=None):
    """Process all PDFs in a directory"""
    if config is None:
        config = load_config()
    
    if directory is None:
        directory = config["watch_dir"]
    
    logger.info(f"Processing directory: {directory}")
    
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                  if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(directory, f))]
    
    results = []
    for pdf_file in pdf_files:
        try:
            output_path = pdf_to_shopify_csv(pdf_file, config=config)
            results.append({"input": pdf_file, "output": output_path, "success": True})
            
            # Move processed file to prevent reprocessing
            processed_dir = os.path.join(directory, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(processed_dir, os.path.basename(pdf_file))
            os.rename(pdf_file, processed_path)
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            results.append({"input": pdf_file, "error": str(e), "success": False})
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PDFs to Shopify-compatible CSVs")
    parser.add_argument("--pdf", help="Path to a single PDF file to convert")
    parser.add_argument("--output", help="Output path for the CSV file")
    parser.add_argument("--dir", help="Directory containing PDFs to convert")
    parser.add_argument("--watch", action="store_true", help="Watch directory for new PDFs")
    
    args = parser.parse_args()
    
    try:
        config = load_config()
        
        if args.pdf:
            pdf_to_shopify_csv(args.pdf, args.output, config)
        
        elif args.dir:
            process_directory(args.dir, config)
        
        elif args.watch:
            import time
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class PDFHandler(FileSystemEventHandler):
                def on_created(self, event):
                    if event.is_directory:
                        return
                    if event.src_path.lower().endswith('.pdf'):
                        logger.info(f"New PDF detected: {event.src_path}")
                        try:
                            pdf_to_shopify_csv(event.src_path, config=config)
                        except Exception as e:
                            logger.error(f"Error processing new PDF: {str(e)}")
            
            watch_dir = config["watch_dir"]
            logger.info(f"Watching directory: {watch_dir}")
            
            event_handler = PDFHandler()
            observer = Observer()
            observer.schedule(event_handler, watch_dir, recursive=False)
            observer.start()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        
        else:
            process_directory(config=config)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
