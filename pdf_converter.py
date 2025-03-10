def process_watch_directory(config=None):
    """
    Monitor the watch directory for new PDF files and process them.

    Args:
        config (dict, optional): Configuration dictionary. If None, will be loaded.
    """
    if not config:
        config = load_config()
    
    watch_dir = config["watch_dir"]
    logger.info(f"Monitoring watch directory: {watch_dir}")
    
    try:
        pdf_files = [f for f in os.listdir(watch_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.info("No PDF files found in watch directory.")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process.")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(watch_dir, pdf_file)
            
            # Extract client name from filename if available
            client_match = re.match(r'^([^_]+)_', pdf_file)
            client = client_match.group(1) if client_match else "default"
            
            # Process the PDF
            try:
                output_path = process_pdf(pdf_path, client, config)
                if output_path:
                    # Move processed file to prevent reprocessing
                    processed_dir = os.path.join(watch_dir, "processed")
                    os.makedirs(processed_dir, exist_ok=True)
                    os.rename(pdf_path, os.path.join(processed_dir, pdf_file))
                    logger.info(f"Moved {pdf_file} to processed directory.")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                # Move failed file to prevent reprocessing attempts
                failed_dir = os.path.join(watch_dir, "failed")
                os.makedirs(failed_dir, exist_ok=True)
                os.rename(pdf_path, os.path.join(failed_dir, pdf_file))
                logger.info(f"Moved {pdf_file} to failed directory.")
    
    except Exception as e:
        logger.error(f"Error monitoring watch directory: {e}")

def main():
    """
    Main function that processes command line arguments or runs in watch mode.
    """
    try:
        config = load_config()
        
        if len(sys.argv) > 1:
            # Process specific PDF file(s)
            for pdf_path in sys.argv[1:]:
                if os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
                    process_pdf(pdf_path, config=config)
                else:
                    logger.error(f"Invalid file path: {pdf_path}")
        else:
            # Run in watch mode
            logger.info("Running in watch mode. Press Ctrl+C to exit.")
            while True:
                process_watch_directory(config)
                logger.info(f"Waiting for new files... (checking every 60 seconds)")
                time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
