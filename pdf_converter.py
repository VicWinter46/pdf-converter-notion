#!/usr/bin/env python3
"""
pdf_converter.py

This script converts PDFs into Shopify-compatible CSVs using advanced text extraction and the Anthropic Claude API.
It extracts text (including vendor hints, image links, and quantity info), sends an enhanced prompt (with the PDF's Base64 string appended)
to Claude, and then parses the CSV output into a final CSV file for Shopify.
"""

import os
import sys
import re
import json
import csv
import time
import base64
import logging
import sqlite3
from datetime import datetime
from io import StringIO
from pathlib import Path

import pdfplumber
import pandas as pd
from PyPDF2 import PdfReader

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_converter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database setup
DB_PATH = "file_metadata.db"

def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY,
            client TEXT,
            filename TEXT,
            status TEXT,
            uploaded_at TEXT,
            processed_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

setup_database()

def load_config():
    """
    Load configuration from config.json or environment variables.
    Creates a default config file if missing.

    Returns:
        dict: Configuration dictionary.
    Raises:
        ValueError: If anthropic_api_key is missing.
    """
    config_path = Path(__file__).parent / "config.json"
    default_config = {
        "anthropic_api_key": "",
        "model": "claude-3-sonnet-20240229",
        "output_dir": "output",
        "watch_dir": "watch",
        "prompt_template": (
            "Extract product details in CSV format:\n"
            "Product Title,Body (HTML),Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color\n\n"
            "If MSRP is missing, use Wholesale Price.\n"
            "Sizes should be standardized (XS, S, M, L, XL, 0-3 M, 2T, etc.).\n"
            "Only include Color if multiple colors exist.\n\n"
            "Return only CSV data."
        )
    }

    # Use environment variable if available.
    if os.getenv("ANTHROPIC_API_KEY"):
        default_config["anthropic_api_key"] = os.getenv("ANTHROPIC_API_KEY")
        logger.info("Using Anthropic API key from environment variable.")

    try:
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as f:
                file_config = json.load(f)
            default_config.update(file_config)
        else:
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            logger.warning(f"Created default config file at {config_path}. Please edit it with your API key.")
    except Exception as e:
        logger.error(f"Error loading config file: {e}")

    if not default_config.get("anthropic_api_key"):
        msg = f"Missing anthropic_api_key in config file. Please add it to {config_path}"
        logger.error(msg)
        raise ValueError(msg)

    os.makedirs(default_config["output_dir"], exist_ok=True)
    os.makedirs(default_config["watch_dir"], exist_ok=True)

    return default_config

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file, including image links and vendor hints.

    Uses PyPDF2 first to extract text and annotations.
    Then enhances extraction with pdfplumber for additional clues (e.g., quantities).

    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Combined extracted text.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        image_links = []
        document_images = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
            # Extract annotations safely:
            annots = page.get("/Annots")
            if annots:
                try:
                    # Resolve indirect objects if needed.
                    if hasattr(annots, "get_object"):
                        annots = annots.get_object()
                    if isinstance(annots, list):
                        for annot in annots:
                            try:
                                obj = annot.get_object() if hasattr(annot, "get_object") else annot
                                if obj.get("/Subtype") == "/Link" and "/A" in obj and "/URI" in obj["/A"]:
                                    uri = obj["/A"]["/URI"]
                                    if uri.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                                        image_links.append(uri)
                            except Exception as ae:
                                logger.debug(f"Annotation error: {ae}")
                    else:
                        logger.debug("Annotations not a list")
                except Exception as ex:
                    logger.debug(f"Error processing annotations: {ex}")

        # Look for potential vendor names in the first 20 lines.
        page_lines = text.split('\n')
        potential_vendors = []
        for line in page_lines[:20]:
            if re.match(r'^[A-Z][A-Z\s]+$', line.strip()) and len(line.strip()) > 3:
                potential_vendors.append(line.strip())

        # Use regex to search for brand/company information.
        brand_section_pattern = r'(?:Brand|Company|Vendor)(?:\s+Information)?[:\s]+([A-Za-z0-9\s]+)'
        brand_matches = re.findall(brand_section_pattern, text, re.IGNORECASE)
        if brand_matches:
            potential_vendors.extend(brand_matches)

        vendor_patterns = [
            r'(?:From|Supplier|Bill\s+from|Sold\s+by|Purchased\s+from)[:\s]+([A-Za-z0-9\s&]+)',
            r'(?:BILL\s+TO|SHIP\s+FROM)[:\s]+([A-Za-z0-9\s&]+)'
        ]
        for pattern in vendor_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                potential_vendors.extend([m.strip() for m in matches if m.strip()])

        for line in page_lines[:5]:
            if line.strip() and not re.search(r'invoice|order|po\s+#|date', line.lower()):
                if len(line.strip()) > 3 and not line.strip().isdigit():
                    potential_vendors.append(line.strip())

        if potential_vendors:
            vendor_text = "\n### POTENTIAL VENDORS ###\n"
            for vendor in potential_vendors:
                vendor_text += f"POTENTIAL VENDOR: {vendor}\n"
            text = vendor_text + text

        # Use pdfplumber to extract image placeholders.
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if hasattr(page, "images") and page.images:
                        for j, _ in enumerate(page.images):
                            document_images.append(f"IMAGE_{i}_{j}")
        except Exception as img_err:
            logger.warning(f"Image extraction error: {img_err}")

        if document_images:
            text += "\n\n### DOCUMENT CONTAINS IMAGES ###\n"
            for img_ref in document_images:
                text += f"IMAGE REFERENCE: {img_ref}\n"

        if image_links:
            text += "\n\n### IMAGE LINKS ###\n"
            for link in image_links:
                text += f"IMAGE URL: {link}\n"

        # Use pdfplumber for extra extraction (e.g., quantities).
        with pdfplumber.open(pdf_path) as pdf:
            plumber_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            size_qty_patterns = [
                r'(XXS|XS|S|M|L|XL|XXL)\s*[:=]?\s*(\d+)',
                r'(XXS|XS|S|M|L|XL|XXL)\s+(\d+)',
                r'Qty\s*[:=]?\s*(\d+)',
                r'Quantity\s*[:=]?\s*(\d+)'
            ]
            quantity_text = "\n### QUANTITY INFORMATION ###\n"
            has_quantities = False
            for pattern in size_qty_patterns:
                qty_matches = re.findall(pattern, plumber_text, re.IGNORECASE)
                if qty_matches:
                    has_quantities = True
                    for match in qty_matches:
                        if isinstance(match, tuple) and len(match) == 2:
                            quantity_text += f"SIZE: {match[0]} QTY: {match[1]}\n"
                        else:
                            quantity_text += f"QTY: {match}\n"
            if has_quantities:
                text += quantity_text

        return text

    except Exception as e:
        logger.error(f"Error in PDF extraction: {e}")
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

def encode_pdf_for_claude(pdf_path):
    """
    Encode the PDF file into a Base64 string.

    Args:
        pdf_path (str): Path to the PDF.
    Returns:
        str: Base64-encoded string.
    Raises:
        Exception: If encoding fails.
    """
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding PDF: {e}")
        raise

def process_with_claude(pdf_path, config):
    """
    Process the PDF using the Anthropic Claude API to extract detailed CSV data.

    Builds an enhanced prompt with vendor guidance and appends the Base64-encoded PDF.
    Uses the new completion.create() method. A monkey-patch is applied to remove any 'proxies'
    keyword argument if present.

    Args:
        pdf_path (str): Path to the PDF.
        config (dict): Configuration dictionary.
    Returns:
        str: CSV-formatted text from Claude.
    Raises:
        Exception: If the API call fails.
    """
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY") or config["anthropic_api_key"]
        extracted_text = extract_text_from_pdf(pdf_path)

        vendor_info = []
        for line in extracted_text.split('\n'):
            if line.startswith("POTENTIAL VENDOR:"):
                vendor = line.replace("POTENTIAL VENDOR:", "").strip()
                vendor_info.append(vendor)

        if vendor_info:
            vendor_guidance = (
                f"- The document mentions these potential vendors: {', '.join(vendor_info)}\n"
                "- Identify the CORRECT vendor for each product\n"
                "- Look at document headers, logos, and branding to determine the true vendor name\n"
                "- Different products may come from different vendors"
            )
        else:
            vendor_guidance = (
                "- Look for the vendor/brand name at the top of the document or in headers\n"
                "- The vendor is typically the company SELLING the products, not the customer\n"
                "- All products in a PO typically come from the same vendor, but verify this"
            )

        pdf_base64 = encode_pdf_for_claude(pdf_path)

        # Import and monkey-patch the Anthropic client to ignore any 'proxies' keyword argument.
        import anthropic
        original_init = anthropic.Client.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.pop("proxies", None)
            return original_init(self, *args, **kwargs)

        anthropic.Client.__init__ = patched_init

        client = anthropic.Client(api_key=api_key)

        enhanced_prompt = (
            "Extract detailed product information from this purchase order for a Shopify import.\n\n"
            "### CRITICAL EXTRACTION RULES:\n"
            "1. VENDOR/BRAND NAME: \n"
            "   - The vendor name is usually at the top of the document\n"
            f"   - {vendor_guidance}\n"
            "   - Do NOT use \"INVOICE\" as the vendor name\n"
            "   - The vendor is the company MAKING the products, not the customer receiving them\n\n"
            "2. SIZE EXTRACTION:\n"
            "   - ONLY extract sizes that have quantities greater than 0\n"
            "   - Look for sections showing sizes and quantities together\n"
            "   - Create a separate row for EACH SIZE with a non-zero quantity\n"
            "   - Skip any sizes that show quantity 0 or no quantity\n"
            "   - Common sizes include: XXS, XS, S, M, L, XL, XXL\n"
            "   - IMPORTANT: Return JUST the size value WITHOUT \"Size \" prefix\n\n"
            "3. TITLE FORMAT:\n"
            "   - DO NOT use dashes in titles, use spaces instead\n"
            "   - Use Title Case For All Words\n"
            "   - Include color in title when available (format: \"in Color\")\n\n"
            "4. SKU AND STYLE NUMBERS:\n"
            "   - Use the exact style/item number as the SKU (e.g., \"336537\", \"342348\")\n"
            "   - Look for numbers following \"Style #\", \"Item #\", or similar labels\n\n"
            "5. IMAGES: \n"
            "   - Extract any image URLs if present in the document\n"
            "   - If no URLs are available, leave the field empty\n\n"
            "### TABLE FORMAT:\n"
            "Product Title,Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color,Product image URL,Quantity\n\n"
            "Return ONLY properly formatted CSV data with header and data rows. No explanations or other text.\n\n"
            "Attached PDF (Base64):\n" + pdf_base64
        )

        response = client.completion.create(
            prompt=enhanced_prompt,
            model=config["model"],
            max_tokens_to_sample=4000,
            temperature=0.1,
            stop_sequences=["\n\nHuman:"]
        )
        result = response.completion

        if result.startswith("```csv") or result.startswith("```"):
            result = re.sub(r'^```csv\s*', '', result)
            result = re.sub(r'^```\s*', '', result)
            result = re.sub(r'\s*```$', '', result)

        if "Product Title" not in result and "Title" not in result:
            logger.warning("First extraction attempt yielded poor results, trying backup prompt.")
            backup_prompt = (
                "Extract all products from this purchase order into a CSV table.\n\n"
                "For each product in the PDF:\n"
                "1. Get the Product Title, Vendor, Product Type, SKU, Price information\n"
                "2. Extract all sizes with quantities > 0\n"
                "3. Include colors if available\n"
                "4. Create a separate row for each size variation\n\n"
                "Return ONLY the CSV with this header (no explanations):\n"
                "Product Title,Vendor,Product Type,SKU,Wholesale Price,MSRP,Size,Color,Product image URL,Quantity\n\n"
                "Attached PDF (Base64):\n" + pdf_base64
            )
            response = client.completion.create(
                prompt=backup_prompt,
                model=config["model"],
                max_tokens_to_sample=4000,
                temperature=0.1,
                stop_sequences=["\n\nHuman:"]
            )
            result = response.completion
            if result.startswith("```csv") or result.startswith("```"):
                result = re.sub(r'^```csv\s*', '', result)
                result = re.sub(r'^```\s*', '', result)
                result = re.sub(r'\s*```$', '', result)

        logger.info("Claude extraction result (first 500 chars): " + (result[:500] + "..." if len(result) > 500 else result))
        return result

    except Exception as e:
        logger.error(f"Error processing with Claude: {e}")
        raise

def parse_csv_data(csv_data):
    """
    Parse CSV text into a pandas DataFrame with multiple fallback methods.

    Args:
        csv_data (str): CSV-formatted text.
    Returns:
        pandas.DataFrame: Parsed DataFrame.
    Raises:
        Exception: If all parsing attempts fail.
    """
    try:
        csv_lines = csv_data.strip().split('\n')
        if not csv_lines:
            raise ValueError("Empty CSV data")
        header = csv_lines[0]
        expected_columns = len(header.split(','))

        fixed_lines = [header]
        for line in csv_lines[1:]:
            line = line.strip()
            if not line:
                continue
            fields = line.split(',')
            if len(fields) > expected_columns:
                fixed_line = []
                field_buffer = []
                count = 0
                for field in fields:
                    field_buffer.append(field)
                    count += 1
                    if count == expected_columns or field == fields[-1]:
                        fixed_line.extend(field_buffer[:-1])
                        if field_buffer:
                            fixed_line.append(" ".join(field_buffer[-1:]))
                        break
                fixed_lines.append(",".join(fixed_line))
            else:
                fixed_lines.append(line)
        clean_csv = "\n".join(fixed_lines)
        logger.info("Cleaned CSV data (first 500 chars): " + (clean_csv[:500] + "..." if len(clean_csv) > 500 else clean_csv))
        try:
            products_df = pd.read_csv(StringIO(clean_csv))
            return products_df
        except Exception as first_error:
            try:
                logger.info("Standard CSV parsing failed, trying quoting options...")
                products_df = pd.read_csv(StringIO(clean_csv), quoting=1)
                return products_df
            except Exception:
                try:
                    logger.info("Trying with on_bad_lines='skip'...")
                    products_df = pd.read_csv(StringIO(clean_csv), on_bad_lines='skip')
                    return products_df
                except Exception:
                    logger.info("Trying manual parsing...")
                    header_fields = fixed_lines[0].split(',')
                    data_rows = []
                    for line in fixed_lines[1:]:
                        fields = line.split(',')
                        # Ensure we have the right number of fields
                        while len(fields) < len(header_fields):
                            fields.append('')
                        if len(fields) > len(header_fields):
                            fields = fields[:len(header_fields)]
                        data_rows.append(dict(zip(header_fields, fields)))
                    
                    if data_rows:
                        return pd.DataFrame(data_rows)
                    else:
                        raise ValueError("Failed to parse CSV data with all methods")
    except Exception as e:
        logger.error(f"Error parsing CSV data: {e}")


import openai

def process_with_openai(text):
    """
    Process extracted text with OpenAI to generate structured Shopify CSV output.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"""
    Extract product details in CSV format:
    Product Title, Vendor, Product Type, SKU, Wholesale Price, MSRP, Size, Color

    - Product Type must be accurately determined from the text.
    - If MSRP is missing, use Wholesale Price * 2.
    - Only include Color if multiple colors exist.
    - Ensure every product has a SKU.
    - Return only the CSV data.

    Purchase Order Text:
    {text}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an AI that structures product data into a Shopify CSV."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error processing with OpenAI: {e}")
        return None

