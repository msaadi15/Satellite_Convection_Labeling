import eumdac
import datetime
import json
import shutil
import zipfile
import os
from tqdm import tqdm
from satpy import Scene
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Credentials
CREDENTIALS_FILE = os.path.join(os.path.expanduser("~"), '.eumetsat_api_key')
COLLECTION_ID = 'EO:EUM:DAT:MSG:HRSEVIRI'
DATA_DIR = "data"


# Target hours for each year
YEAR_CONFIG = {
    2023: [3, 6, 9, 12, 15, 18, 21, 24],
    2024: [12, 15, 18]
}

# Region for cropping
MOROCCO_BBOX = {"lon_min": -20, "lon_max": 5, "lat_min": 20, "lat_max": 40.0}

# =============================================================================
# INITIALIZATION
# =============================================================================

def setup_eumetsat():
    """Initialize EUMETSAT Data Store connection"""
    print("🔐 Connecting to EUMETSAT Data Store...")
    
    with open(CREDENTIALS_FILE) as f:
        creds = json.load(f)
    
    token = eumdac.AccessToken((creds['consumer_key'], creds['consumer_secret']))
    return eumdac.DataStore(token).get_collection(COLLECTION_ID)

def generate_dates():
    """Generate all dates to process based on year configuration"""
    dates = []
    for year, hours in YEAR_CONFIG.items():
        for month in range(1, 13):
            # Days in month calculation
            if month in [1, 3, 5, 7, 8, 10, 12]:
                days = 31
            elif month in [4, 6, 9, 11]:
                days = 30
            else:  # February
                days = 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28
            
            for day in range(1, days + 1):
                dates.append((year, month, day, hours))
    
    return dates

# =============================================================================
# NAT FILE PROCESSING FUNCTION
# =============================================================================

def process_nat_to_image(nat_file_path: str, output_dir: str) -> bool:
    """
    Process a .nat file to create RGB image and remove the original file
    """
    try:
        # Extract timestamp from filename
        filename = os.path.basename(nat_file_path)
        parts = filename.split('-')
        
        # Find the timestamp part
        timestamp_part = None
        for part in parts:
            if part.startswith('202') and part.endswith('Z'):
                timestamp_part = part
                break
        
        if timestamp_part:
            # Convert to ISO format
            date_str = timestamp_part[:8]  # YYYYMMDD
            time_str = timestamp_part[8:14]  # HHMMSS
            millis_str = timestamp_part[15:18] if '.' in timestamp_part and len(timestamp_part) > 15 else '000'
            
            iso_filename = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[:2]}-{time_str[2:4]}-{time_str[4:6]}.{millis_str}Z.png"
        else:
            iso_filename = f"{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')}.000Z.png"
        
        output_path = os.path.join(output_dir, iso_filename)
        
        # Load and process the .nat file
        scn = Scene(filenames=[nat_file_path], reader="seviri_l1b_native")
        scn.load(["IR_108", "IR_039", "IR_016", "VIS006", "WV_062", "WV_073"])
        
        # Crop to Morocco
        cropped_scn = scn.crop(ll_bbox=(MOROCCO_BBOX["lon_min"], MOROCCO_BBOX["lat_min"], 
                                  MOROCCO_BBOX["lon_max"], MOROCCO_BBOX["lat_max"]))
        
        
                # Get the convection data
        conv_data = cropped_scn['convection'].values
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot with appropriate colormap
        img = ax.imshow(conv_data, cmap='RdYlBu_r')  # Good for convection
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax, shrink=0.8)
        cbar.set_label('Convection Intensity', fontsize=12)
        
        # Add title and labels
        ax.set_title('SEVIRI Convection Analysis', fontsize=16, pad=20)
        ax.axis('off')  # Turn off axes for cleaner image
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"✅ Convection image saved: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error processing .nat file: {str(e)}")
        return False

# =============================================================================
# DOWNLOAD FUNCTIONS (FIXED)
# =============================================================================

def download_and_process_product(product, data_dir):
    """Download and process a single product using the correct method"""
    try:
        # Use product.open() to get the file stream and filename
        with product.open() as fsrc:
            # Get the filename from the stream
            filename = fsrc.name
            file_path = os.path.join(data_dir, filename)
            
            # Download the file
            with open(file_path, 'wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)
            
            print(f'📥 Downloaded: {filename}')
            
            # Check if it's a ZIP file
            if file_path.endswith('.zip'):
                print(f'📦 Extracting: {filename}')
                
                # Extract the ZIP file
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                
                print(f'✅ Extracted: {filename}')
                
                # Remove the ZIP file after extraction
                os.remove(file_path)
                print(f'🗑️  Removed ZIP: {filename}')
                
                # Process all .nat files that were extracted
                extracted_files = [f for f in os.listdir(data_dir) if f.endswith('.nat') and f not in filename]
                for nat_file in extracted_files:
                    nat_file_path = os.path.join(data_dir, nat_file)
                    process_nat_to_image(nat_file_path, data_dir)
            
            else:
                # If it's already a .nat file, process it directly
                process_nat_to_image(file_path, data_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing product: {str(e)}")
        # Clean up any partially downloaded files
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main download and processing workflow"""
    # Setup
    collection = setup_eumetsat()
    os.makedirs(DATA_DIR, exist_ok=True)
    
    dates = generate_dates()
    print(f"📅 Processing {len(dates)} days across {len(YEAR_CONFIG)} years")
    print(f"📁 Images will be saved to: {DATA_DIR}")
    print(f"🎯 Target region: {MOROCCO_BBOX}")
    
    total_downloaded = 0
    total_processed = 0
    
    # Main progress loop
    for year, month, day, target_hours in tqdm(dates, desc="Overall progress", unit="day"):
        date_str = f"{year}-{month:02d}-{day:02d}"
        
        try:
            # Process each target hour
            for hour in target_hours:
                # Adjust hour for 24-hour format
                
                
                # Create time window
                start = datetime.datetime(year, month, day, hour - 1, 55)
                end = datetime.datetime(year, month, day, hour-1, 59)
                
                # Search for products
                products = list(collection.search(dtstart=start, dtend=end))
                
                if not products:
                    continue
                
                # Download and process each product
                for product in products:
                    if download_and_process_product(product, DATA_DIR):
                        total_downloaded += 1
                        total_processed += 1
                        
        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")
            continue
    
    # Final cleanup and summary
    print(f"\n✅ Processing completed!")
    print(f"📊 Total products downloaded: {total_downloaded}")
    print(f"🖼️  Total images created: {total_processed}")
    
    # Check for any remaining .nat files (should be none)
    remaining_nat_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.nat')]
    if remaining_nat_files:
        print(f"⚠️  Warning: {len(remaining_nat_files)} .nat files remaining (processing failed)")
        for file in remaining_nat_files:
            print(f"   - {file}")
    
    # Show final image count
    image_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
    print(f"📁 Final image count: {len(image_files)}")
    print(f"💾 All images saved in: {DATA_DIR}")

if __name__ == "__main__":
    main()