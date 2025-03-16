import pdfplumber
import openai
import pandas as pd
import os
import json
import logging
import sys
import re
from io import StringIO
from datetime import datetime
from PyPDF2 import PdfReader

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
    """Load configuration from config.json file or environment variables"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    
    # Start with default config
    default_config = {
        "openai_api_key": "",
        "model": "gpt-3.5-turbo",
        "output_dir": "output",
        "watch_dir": "watch",
        "prompt_template": "Extract product details in CSV format:\nProduct Title,Body (HTML),Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color\n\nIf MSRP is missing, use Wholesale Price.\nSizes should be standardized (XS, S, M, L, XL, 0-3 M, 2T, etc.).\nOnly include Color if multiple colors exist.\n\nText:\n{text}\n\nReturn only CSV data."
    }
    
    # Check for environment variables first
    if os.getenv("OPENAI_API_KEY"):
        default_config["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        logger.info("Using OpenAI API key from environment variable")
    
    # Try to load from config file
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update default config with file config
            for key, value in config.items():
                default_config[key] = value
        else:
            # Create default config file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            logger.warning(f"Created default config file at {config_path}. Please edit it with your API key.")
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
    
    # Final check for required fields
    if not default_config.get("openai_api_key"):
        logger.error(f"Missing openai_api_key in config file. Please add it to {config_path}")
        raise ValueError("Missing openai_api_key in config file")
    
    # Create directories if they don't exist
    os.makedirs(default_config["output_dir"], exist_ok=True)
    os.makedirs(default_config["watch_dir"], exist_ok=True)
    
    return default_config

def extract_text_from_pdf(pdf_path):
    """Extract text and hyperlinks from a PDF file with improved image URL extraction"""
    try:
        # First try to extract with PyPDF2 to get hyperlinks
        reader = PdfReader(pdf_path)
        text = ""
        image_links = []
        document_images = []
        
        # Extract text and find links
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
            
            # Extract annotations that might contain links
            if '/Annots' in page:
                for annot in page['/Annots']:
                    obj = annot.get_object()
                    if obj['/Subtype'] == '/Link' and '/A' in obj and '/URI' in obj['/A']:
                        uri = obj['/A']['/URI']
                        
                        # Check if it's an image link
                        if uri.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                            image_links.append(uri)
        
        # Look for potential brand logos or main vendor name at the top of the document
        page_lines = text.split('\n')
        potential_vendors = []
        
        # Look for lines with capitalized text that might be vendor names
        for line in page_lines[:20]:  # Check first 20 lines
            # Look for all-caps words that might be brand names
            if re.match(r'^[A-Z][A-Z\s]+$', line.strip()) and len(line.strip()) > 3:
                potential_vendors.append(line.strip())
        
        # Also look for brand information sections
        brand_section_pattern = r'(?:Brand|Company|Vendor)(?:\s+Information)?[:\s]+([A-Za-z0-9\s]+)'
        brand_matches = re.findall(brand_section_pattern, text, re.IGNORECASE)
        if brand_matches:
            potential_vendors.extend(brand_matches)
        
        # Add vendor detection markers to help the AI
        if potential_vendors:
            vendor_text = "\n### POTENTIAL VENDORS ###\n"
            for vendor in potential_vendors:
                vendor_text += f"POTENTIAL VENDOR: {vendor.strip()}\n"
            text = vendor_text + text
        
        # Try to extract images from the PDF
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract images if available
                    if hasattr(page, 'images') and page.images:
                        for j, img in enumerate(page.images):
                            document_images.append(f"IMAGE_{i}_{j}")
        except Exception as img_err:
            logger.warning(f"Image extraction error: {str(img_err)}")
        
        # Add image presence markers to help OpenAI understand there are images
        if document_images:
            text += "\n\n### DOCUMENT CONTAINS IMAGES ###\n"
            for img_ref in document_images:
                text += f"IMAGE REFERENCE: {img_ref}\n"
        
        # Add image links to the text so OpenAI can find them
        if image_links:
            text += "\n\n### IMAGE LINKS ###\n"
            for link in image_links:
                text += f"IMAGE URL: {link}\n"
        
        # Use enhanced extraction with pdfplumber to find quantity information
        with pdfplumber.open(pdf_path) as pdf:
            plumber_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            
            # Look for size/quantity patterns like "XS: 1" or similar
            size_qty_patterns = [
                r'(XXS|XS|S|M|L|XL|XXL)\s*[:=]?\s*(\d+)',  # XS: 1
                r'(XXS|XS|S|M|L|XL|XXL)\s+(\d+)',          # XS 1
                r'Qty\s*[:=]?\s*(\d+)',                    # Qty: 1
                r'Quantity\s*[:=]?\s*(\d+)'                # Quantity: 1
            ]
            
            # Add special markers for quantities detected
            quantity_text = "\n### QUANTITY INFORMATION ###\n"
            has_quantities = False
            
            for pattern in size_qty_patterns:
                qty_matches = re.findall(pattern, plumber_text, re.IGNORECASE)
                if qty_matches:
                    has_quantities = True
                    for match in qty_matches:
                        if len(match) == 2:  # Size and quantity
                            quantity_text += f"SIZE: {match[0]} QTY: {match[1]}\n"
                        else:  # Just quantity
                            quantity_text += f"QTY: {match[0]}\n"
            
            if has_quantities:
                text += quantity_text
        
        return text
        
    except Exception as e:
        logger.error(f"Error in PDF extraction: {str(e)}")
        # Fall back to regular text extraction
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text

def process_with_openai(text, config):
    """Process text with OpenAI API - with improved vendor and sizing extraction"""
    try:
        # Set API key from environment or config
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai.api_key = config["openai_api_key"]
            
        # Look for potential vendor information
        vendor_info = []
        for line in text.split('\n'):
            if line.startswith("POTENTIAL VENDOR:"):
                vendor = line.replace("POTENTIAL VENDOR:", "").strip()
                vendor_info.append(vendor)
        
        vendor_guidance = ""
        if vendor_info:
            vendor_guidance = f"""
- The document mentions these potential vendors: {', '.join(vendor_info)}
- Identify the CORRECT vendor for each product
- Look at document headers, logos, and branding to determine the true vendor name
- Different products may come from different vendors"""
        else:
            vendor_guidance = """
- Look for the vendor/brand name at the top of the document or in headers
- The vendor is typically the company SELLING the products, not the customer
- All products in a PO typically come from the same vendor, but verify this"""
        
        # Improved prompt with better vendor detection and exact sizing extraction
        enhanced_prompt = f"""
Extract detailed product information from this purchase order for a Shopify import.

### CRITICAL EXTRACTION RULES:
1. VENDOR/BRAND NAME: {vendor_guidance}

2. SIZE EXTRACTION:
   - ONLY extract sizes that have quantities greater than 0
   - Look for sections showing sizes and quantities together
   - Create a separate row for EACH SIZE with a non-zero quantity
   - Skip any sizes that show quantity 0 or no quantity
   - Common sizes include: XXS, XS, S, M, L, XL, XXL

3. SKU AND STYLE NUMBERS:
   - Use the exact style/item number as the SKU (e.g., "336537", "342348")
   - Look for numbers following "Style #", "Item #", or similar labels

4. IMAGES: 
   - Extract any image URLs if present in the document
   - If no URLs are available, leave the field empty

5. PRODUCT TITLE:
   - Capitalize each word in the product title (e.g., "summer dress" should become "Summer Dress")

### TABLE FORMAT:
Product Title,Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color,Product image URL,Quantity

DOCUMENT TEXT:
{text}

Return ONLY properly formatted CSV data with header and data rows. No explanations or other text."""

        # Use OpenAI API with specialized prompt
        response = openai.ChatCompletion.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a specialized purchase order extraction system that accurately identifies vendors, products, sizes with quantities, and pricing."},
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.1  # Lower temperature for more consistent results
        )
        
        # Get result and verify it contains data
        result = response['choices'][0]['message']['content']
        if "Product Title" not in result or "," not in result or len(result.split('\n')) < 2:
            # Try one more time with different prompt if extraction failed
            logger.warning("First extraction attempt failed, trying again with alternate prompt")
            
            backup_prompt = f"""
Extract product information from this purchase order with EXACT quantities for each size.

IMPORTANT:
1. For each product, identify:
   - Full product name and description
   - Brand/vendor name (the company selling the items, not the customer)
   - Product type (Dress, Blouse, etc.)
   - SKU/Style number 
   - Prices (wholesale and retail)
   - ONLY sizes with quantities greater than 0
   - Colors

2. Format each size as a separate row in the CSV
   - If a product comes in sizes XS, S, M and quantities are XS:1, S:2, M:0
   - Only include rows for XS and S (not M since quantity is 0)

3. Carefully identify the correct vendor name from the document header or branding

4. Capitalize each word in the product title (e.g., "summer dress" should become "Summer Dress")

Here's the purchase order to analyze:
{text}

Return ONLY the CSV with this header (include Quantity column):
Product Title,Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color,Product image URL,Quantity"""
            
            response = openai.ChatCompletion.create(
                model=config["model"],
                messages=[{"role": "user", "content": backup_prompt}],
                temperature=0.1
            )
            result = response['choices'][0]['message']['content']
            
        # Log result for debugging
        logger.info("Extraction result:")
        logger.info(result)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing with OpenAI: {str(e)}")
        raise

def format_for_shopify(products_df):
    """Format the DataFrame for Shopify import with improved handling"""
    try:
        # Apply post-processing to fix common issues
        products_df = post_process_data(products_df)
        
        # Print column names for debugging
        logger.info(f"DataFrame columns: {products_df.columns.tolist()}")
        
        # Create new DataFrame for Shopify with the required structure
        shopify_data = []

        # Iterate through products to create main and variant rows
        for idx, product in products_df.iterrows():
            # Main product row
            main_row = {
                "Title": product["Product Title"],
                "URL handle": product["Product Title"].lower().replace(" ", "-"),
                "Description": product.get("Description", ""),
                "Vendor": product["Vendor"],
                "Type": product["Product Type"],
                "Status": "active",
                "Option1 name": "Size" if "Size" in product else "Title",
                "Option1 value": product.get("Size", product["Product Title"]),
                "Product image URL": product.get("Product image URL", ""),
                "SKU": "",
                "Price": "",
                "Cost per item": "",
                "Inventory tracker": "",
                "Inventory quantity": "",
                "Option2 name": "",
                "Option2 value": "",
                "Continue selling when out of stock": ""
            }
            shopify_data.append(main_row)

            # Variant rows if there are additional sizes/colors
            if "Size" in product or "Color" in product:
                variant_row = {
                    "Title": "",
                    "URL handle": product["Product Title"].lower().replace(" ", "-"),
                    "Description": "",
                    "Vendor": "",
                    "Type": "",
                    "Status": "",
                    "SKU": product["SKU"],
                    "Option1 name": "Size" if "Size" in product else "",
                    "Option1 value": product.get("Size", ""),
                    "Option2 name": "Color" if "Color" in product else "",
                    "Option2 value": product.get("Color", ""),
                    "Price": product["MSRP"] if pd.notna(product["MSRP"]) else product["Wholesale Price"] * 2,
                    "Cost per item": product["Wholesale Price"],
                    "Inventory tracker": "shopify",
                    "Inventory quantity": product["Quantity"],
                    "Continue selling when out of stock": "deny",
                    "Product image URL": ""
                }
                shopify_data.append(variant_row)

        # Convert list of dictionaries to DataFrame
        shopify_df = pd.DataFrame(shopify_data)

        # Ensure all required columns are present
        required_columns = [
            "Title", "URL handle", "Description", "Vendor", "Type",
            "Status", "SKU", "Option1 name", "Option1 value",
            "Option2 name", "Option2 value", "Price", "Cost per item",
            "Inventory tracker", "Inventory quantity", "Continue selling when out of stock", "Product image URL"
        ]
        for col in required_columns:
            if col not in shopify_df.columns:
                shopify_df[col] = ""

        # Log DataFrame information
        logger.info(f"Final DataFrame columns: {shopify_df.columns.tolist()}")
        logger.info(f"DataFrame shape: {shopify_df.shape}")
        
        return shopify_df
    
    except Exception as e:
        logger.error(f"Error formatting for Shopify: {str(e)}")
        # Print more debug info
        logger.error(f"DataFrame info: {type(products_df)}")
        if isinstance(products_df, pd.DataFrame):
            logger.error(f"DataFrame columns: {products_df.columns.tolist()}")
            logger.error(f"DataFrame sample: {products_df.head().to_dict()}")
        raise

def format_for_shopify(products_df):
    """Format the DataFrame for Shopify import with improved handling"""
    try:
        # Apply post-processing to fix common issues
        products_df = post_process_data(products_df)
        
        # Print column names for debugging
        logger.info(f"DataFrame columns: {products_df.columns.tolist()}")
        
        # Ensure required columns exist
        required_columns = ["Product Title", "Vendor", "Product Type", "SKU", "Wholesale Price", "MSRP", "Size", "Color"]
        for col in required_columns:
            if col not in products_df.columns:
                products_df[col] = None
                logger.info(f"Added missing column: {col}")
        
        # Clean up price values and ensure consistent formatting
        for price_col in ["Wholesale Price", "MSRP"]:
            if price_col in products_df.columns:
                # Convert any price strings to numeric values
                products_df[price_col] = products_df[price_col].apply(
                    lambda x: float(str(x).replace('USD', '').replace('$', '').strip()) if pd.notna(x) and str(x).strip() != '' else None
                )
        
        # Calculate price (use MSRP if available, otherwise use Wholesale Price * 2)
        def calculate_price(row):
            try:
                if "MSRP" in row and pd.notna(row["MSRP"]) and row["MSRP"] != 0:
                    return float(row["MSRP"])
                elif "Wholesale Price" in row and pd.notna(row["Wholesale Price"]):
                    return float(row["Wholesale Price"]) * 2
                else:
                    return None
            except:
                return None
        
        # Create new DataFrame with Shopify columns
        shopify_df = pd.DataFrame()
        
        # Map columns to Shopify format - with flexible naming
        column_mapping = {
            "Product Title": "Title",
            "Title": "Title",  # In case it's already named "Title"
            "SKU": "SKU", 
            "Vendor": "Vendor",
            "Product Type": "Type",
            "Type": "Type",  # In case it's already named "Type"
            "Wholesale Price": "Cost per item",
            "Cost": "Cost per item",  # Alternative naming
            "MSRP": "Price",
            "Price": "Price",  # In case it's already named "Price"
            "Size": "Option1 value",
            "Color": "Option2 value",
            "Product image URL": "Product image URL"
        }
        
        # Map existing columns
        for old_col, new_col in column_mapping.items():
            if old_col in products_df.columns:
                shopify_df[new_col] = products_df[old_col]
                logger.info(f"Mapped {old_col} to {new_col}")
        
        # Set default values for required fields
        if "Title" in shopify_df.columns:
            shopify_df["URL handle"] = shopify_df["Title"].astype(str).str.lower().str.replace(' ', '-').str.replace('[^a-z0-9-]', '', regex=True)
            shopify_df["Description"] = shopify_df["Title"]  # As requested, just use title for description
        else:
            logger.error("Title column not found in DataFrame")
            shopify_df["URL handle"] = ""
            shopify_df["Description"] = ""
        
        shopify_df["Status"] = "draft"  # Set status to draft
        shopify_df["Option1 name"] = "Size"  # Default option name
        shopify_df["Option2 name"] = "Color"  # Default option name
        shopify_df["Continue selling when out of stock"] = "deny"  # Change to deny
        
        # Make sure we have a price
        if "Price" not in shopify_df.columns and "Cost per item" in shopify_df.columns:
            # If no MSRP was found, use Cost per item * 2
            shopify_df["Price"] = shopify_df["Cost per item"].apply(
                lambda x: float(x) * 2 if pd.notna(x) and x != '' else None
            )
        
        # Select columns for Shopify
        shopify_columns = [
            "Title", "URL handle", "Description", "Vendor", "Type", 
            "Status", "SKU", "Option1 name", "Option1 value", 
            "Option2 name", "Option2 value", "Price", "Cost per item",
            "Continue selling when out of stock", "Product image URL"
        ]
        
        # For any missing columns in the requested list, add them with empty values
        for col in shopify_columns:
            if col not in shopify_df.columns:
                shopify_df[col] = ""
                logger.info(f"Added empty column: {col}")
        
        # Log DataFrame information
        logger.info(f"Final DataFrame columns: {shopify_df.columns.tolist()}")
        logger.info(f"DataFrame shape: {shopify_df.shape}")
        
        # Create final CSV
        shopify_csv = shopify_df[shopify_columns]
        return shopify_csv
    
    except Exception as e:
        logger.error(f"Error formatting for Shopify: {str(e)}")
        # Print more debug info
        logger.error(f"DataFrame info: {type(products_df)}")
        if isinstance(products_df, pd.DataFrame):
            logger.error(f"DataFrame columns: {products_df.columns.tolist()}")
            logger.error(f"DataFrame sample: {products_df.head().to_dict()}")
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
                    try:
                        # For older pandas versions
                        products_df = pd.read_csv(StringIO(clean_csv), error_bad_lines=False)
                    except TypeError:
                        # For newer pandas versions
                        products_df = pd.read_csv(StringIO(clean_csv), on_bad_lines='skip')
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
                            header_fields = [field.strip('"').strip() for field in header_fields]
                            data = []
                            
                            for i in range(1, len(fixed_lines)):
                                row = {}
                                fields = fixed_lines[i].split(',')
                                fields = [field.strip('"').strip() for field in fields]
                                
                                # Match available fields to header
                                for j in range(min(len(header_fields), len(fields))):
                                    row[header_fields[j]] = fields[j]
                                    
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

def post_process_data(products_df):
    """Apply additional post-processing to fix common issues"""
    try:
        # Ensure sizes are standardized and precise
        if "Size" in products_df.columns:
            # Map to standardized sizes
            size_mapping = {
                "XSMALL": "XS",
                "SMALL": "S",
                "MEDIUM": "M",
                "LARGE": "L",
                "XLARGE": "XL",
                "XXSMALL": "XXS",
                "XXLARGE": "XXL",
            }
            
            for idx, row in products_df.iterrows():
                size = str(row["Size"]).strip().upper()
                if size in size_mapping:
                    products_df.at[idx, "Size"] = size_mapping[size]
        
        # Make sure each row represents a unique product-size combination
        # This is essential for Shopify imports
        if "Product Title" in products_df.columns and "Size" in products_df.columns and "SKU" in products_df.columns:
            # Check for duplicate product-size combinations
            products_df["_temp_key"] = products_df["SKU"] + ":" + products_df["Size"].astype(str)
            
            # If we have duplicates, only keep one row per combination
            if products_df["_temp_key"].duplicated().any():
                logger.warning("Found duplicate product-size combinations, removing duplicates")
                products_df = products_df.drop_duplicates(subset="_temp_key")
            
            # Remove temporary column
            products_df = products_df.drop(columns=["_temp_key"])
        
        # Check for common issues with vendor names
        if "Vendor" in products_df.columns:
            # Get unique vendors
            vendors = products_df["Vendor"].unique()
            
            # Fix obvious vendor errors - vendor should not have size or color in name
            size_pattern = r'(XXS|XS|S|M|L|XL|XXL)'
            
            for idx, row in products_df.iterrows():
                vendor = row["Vendor"]
                
                # Fix vendor names that accidentally include size info
                if re.search(size_pattern, vendor):
                    # Try to get the part before the size
                    parts = re.split(size_pattern, vendor)
                    if parts and parts[0].strip():
                        products_df.at[idx, "Vendor"] = parts[0].strip()
        
        # Handle product images
        if "Product image URL" in products_df.columns:
            # If there are any valid image URLs, use the first one for all rows with the same SKU
            valid_urls = {}  # SKU -> URL mapping
            
            # First pass - collect valid URLs by SKU
            for idx, row in products_df.iterrows():
                if "SKU" in row and pd.notna(row["SKU"]) and "Product image URL" in row:
                    url = row["Product image URL"]
                    if url and str(url).strip():
                        valid_urls[row["SKU"]] = url
            
            # Second pass - apply URLs to all rows with same SKU
            for idx, row in products_df.iterrows():
                if "SKU" in row and pd.notna(row["SKU"]):
                    sku = row["SKU"]
                    if sku in valid_urls:
                        products_df.at[idx, "Product image URL"] = valid_urls[sku]
        
        # Final check for 0 quantity items - remove them if they exist
        if "Quantity" in products_df.columns:
            logger.info("Removing products with 0 quantity")
            # Convert quantity to numeric first
            products_df["Quantity"] = pd.to_numeric(products_df["Quantity"], errors='coerce')
            products_df = products_df[products_df["Quantity"] > 0]
            
        return products_df
    except Exception as e:
        logger.error(f"Error in post-processing: {str(e)}")
        return products_df  # Return original dataframe if post-processing fails

def format_for_shopify(products_df):
    """Format the DataFrame for Shopify import with improved handling"""
    try:
        # Apply post-processing to fix common issues
        products_df = post_process_data(products_df)
        
        # Print column names for debugging
        logger.info(f"DataFrame columns: {products_df.columns.tolist()}")
        
        # Ensure required columns exist
        required_columns = ["Product Title", "Vendor", "Product Type", "SKU", "Wholesale Price", "MSRP", "Size", "Color"]
        for col in required_columns:
            if col not in products_df.columns:
                products_df[col] = None
                logger.info(f"Added missing column: {col}")
        
        # Clean up price values and ensure consistent formatting
        for price_col in ["Wholesale Price", "MSRP"]:
            if price_col in products_df.columns:
                # Convert any price strings to numeric values
                products_df[price_col] = products_df[price_col].apply(
                    lambda x: float(str(x).replace('USD', '').replace('$', '').strip()) if pd.notna(x) and str(x).strip() != '' else None
                )
        
        # Calculate price (use MSRP if available, otherwise use Wholesale Price * 2)
        def calculate_price(row):
            try:
                if "MSRP" in row and pd.notna(row["MSRP"]) and row["MSRP"] != 0:
                    return float(row["MSRP"])
                elif "Wholesale Price" in row and pd.notna(row["Wholesale Price"]):
                    return float(row["Wholesale Price"]) * 2
                else:
                    return None
            except:
                return None
        
        # Create new DataFrame with Shopify columns
        shopify_df = pd.DataFrame()
        
        # Map columns to Shopify format - with flexible naming
        column_mapping = {
            "Product Title": "Title",
            "Title": "Title",  # In case it's already named "Title"
            "SKU": "SKU", 
            "Vendor": "Vendor",
            "Product Type": "Type",
            "Type": "Type",  # In case it's already named "Type"
            "Wholesale Price": "Cost per item",
            "Cost": "Cost per item",  # Alternative naming
            "MSRP": "Price",
            "Price": "Price",  # In case it's already named "Price"
            "Size": "Option1 value",
            "Color": "Option2 value",
            "Product image URL": "Product image URL"
        }
        
        # Map existing columns
        for old_col, new_col in column_mapping.items():
            if old_col in products_df.columns:
                shopify_df[new_col] = products_df[old_col]
                logger.info(f"Mapped {old_col} to {new_col}")
        
        # Set default values for required fields
        if "Title" in shopify_df.columns:
            shopify_df["URL handle"] = shopify_df["Title"].astype(str).str.lower().str.replace(' ', '-').str.replace('[^a-z0-9-]', '', regex=True)
            shopify_df["Description"] = shopify_df["Title"]  # As requested, just use title for description
        else:
            logger.error("Title column not found in DataFrame")
            shopify_df["URL handle"] = ""
            shopify_df["Description"] = ""
        
        shopify_df["Status"] = "draft"  # Set status to draft
        shopify_df["Option1 name"] = "Size"  # Default option name
        shopify_df["Option2 name"] = "Color"  # Default option name
        shopify_df["Continue selling when out of stock"] = "DENY"
        
        # Make sure we have a price
        if "Price" not in shopify_df.columns and "Cost per item" in shopify_df.columns:
            # If no MSRP was found, use Cost per item * 2
            shopify_df["Price"] = shopify_df["Cost per item"].apply(
                lambda x: float(x) * 2 if pd.notna(x) and x != '' else None
            )
        
        # Select columns for Shopify
        shopify_columns = [
            "Title", "URL handle", "Description", "Vendor", "Type", 
            "Status", "SKU", "Option1 name", "Option1 value", 
            "Option2 name", "Option2 value", "Price", "Cost per item",
            "Continue selling when out of stock", "Product image URL"
        ]
        
        # For any missing columns in the requested list, add them with empty values
        for col in shopify_columns:
            if col not in shopify_df.columns:
                shopify_df[col] = ""
                logger.info(f"Added empty column: {col}")
        
        # Log DataFrame information
        logger.info(f"Final DataFrame columns: {shopify_df.columns.tolist()}")
        logger.info(f"DataFrame shape: {shopify_df.shape}")
        
        # Create final CSV
        shopify_csv = shopify_df[shopify_columns]
        return shopify_csv
    
    except Exception as e:
        logger.error(f"Error formatting for Shopify: {str(e)}")
        # Print more debug info
        logger.error(f"DataFrame info: {type(products_df)}")
        if isinstance(products_df, pd.DataFrame):
            logger.error(f"DataFrame columns: {products_df.columns.tolist()}")
            logger.error(f"DataFrame sample: {products_df.head().to_dict()}")
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
