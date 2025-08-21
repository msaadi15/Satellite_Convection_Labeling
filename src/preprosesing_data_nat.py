from satpy import Scene
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Dict, Tuple

def process_seviri_nat_file(nat_file_path: str, 
                           output_dir: str = None,
                           channels: List[str] = None,
                           bbox: Dict[str, float] = None,
                           show_plot: bool = False,
                           save_image: bool = True,
                           dpi: int = 300) -> Optional[np.ndarray]:
    """
    Process a SEVIRI .nat file to create a BTD RGB composite image.
    
    Args:
        nat_file_path (str): Path to the .nat file
        output_dir (str): Directory to save output image. If None, uses same directory as input file.
        channels (List[str]): List of channels to load. Default: ["IR_108", "IR_039", "IR_016", "VIS006", "WV_062", "WV_073"]
        bbox (Dict[str, float]): Bounding box for cropping. Default: Morocco region
        show_plot (bool): Whether to display the plot
        save_image (bool): Whether to save the image to file
        dpi (int): DPI for saved image
        
    Returns:
        np.ndarray: RGB image array (height, width, 3) or None if failed
        
    Raises:
        FileNotFoundError: If the .nat file doesn't exist
        ValueError: If required channels are not available
    """
    
    # Default channels for BTD RGB composite
    if channels is None:
        channels = ["IR_108", "IR_039", "IR_016", "VIS006", "WV_062", "WV_073"]
    
    # Default bounding box (Morocco region)
    if bbox is None:
        bbox = {"lon_min": -20, "lon_max": 5, "lat_min": 20, "lat_max": 40.0}
    
    # Check if file exists
    if not os.path.exists(nat_file_path):
        raise FileNotFoundError(f"File not found: {nat_file_path}")
    
    try:
        # Load scene
        print(f"📖 Loading scene: {os.path.basename(nat_file_path)}")
        scn = Scene(filenames=[nat_file_path], reader="seviri_l1b_native")
        
        # Load required channels
        print(f"📡 Loading channels: {', '.join(channels)}")
        scn.load(channels)
        
        # Crop to specified region
        print(f"✂️  Cropping to region: {bbox}")
        cropped_scn = scn.crop(ll_bbox=(bbox["lon_min"], bbox["lat_min"], 
                                      bbox["lon_max"], bbox["lat_max"]))
        
        # Extract data as NumPy arrays
        print("🔍 Extracting channel data...")
        bt_wv62 = cropped_scn["WV_062"].values
        bt_wv73 = cropped_scn["WV_073"].values
        bt_ir39 = cropped_scn["IR_039"].values
        bt_ir108 = cropped_scn["IR_108"].values
        refl_nir16 = cropped_scn["IR_016"].values
        refl_vis06 = cropped_scn["VIS006"].values
        
        # Calculate differences for RGB composite
        print("🎨 Creating RGB composite...")
        red_channel = bt_wv62 - bt_wv73    # WV6.2 - WV7.3
        green_channel = bt_ir39 - bt_ir108 # IR3.9 - IR10.8
        blue_channel = refl_nir16 - refl_vis06 # NIR1.6 - VIS0.6
                plt.title(f"SEVIRI RGB Composite - {iso_filename.replace('.png', '')}", fontsize=14, pad=20)

        # Normalize channels (0-1 range)
        def normalize(arr, vmin=None, vmax=None):
            if vmin is None:
                vmin = np.nanpercentile(arr, 2)
            if vmax is None:
                vmax = np.nanpercentile(arr, 98)
            arr_clipped = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
            return np.nan_to_num(arr_clipped)
        
        R = normalize(red_channel)
        G = normalize(green_channel)
        B = normalize(blue_channel)
        
        # Stack channels into RGB image
        rgb = np.dstack((R, G, B))
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.dirname(nat_file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        if save_image:
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(nat_file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_RGB.png")
            
            # Save the image
            plt.figure(figsize=(12, 10))
            plt.imshow(np.rot90(rgb, k=2))
            plt.axis('off')
            #plt.title(f"BTD RGB Composite - {base_name}", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            print(f"💾 Image saved: {output_path}")
        
        if show_plot:
            # Display the image
            fig, ax = plt.subplots(figsize=(12, 10))
            img = ax.imshow(np.rot90(rgb, k=2))
            ax.set_title(f"BTD RGB Composite\n{os.path.basename(nat_file_path)}", 
                        fontsize=16, pad=20)
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(img, ax=ax, shrink=0.8)
            cbar.set_label("Normalized Brightness Temperature Difference", fontsize=12)
            
            # Add bounding box info
            bbox_text = (f"Region: {bbox['lon_min']}°E to {bbox['lon_max']}°E, "
                       f"{bbox['lat_min']}°N to {bbox['lat_max']}°N")
            plt.figtext(0.5, 0.01, bbox_text, ha='center', fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
        
        print("✅ Processing completed successfully!")
        return rgb
        
    except Exception as e:
        print(f"❌ Error processing {nat_file_path}: {str(e)}")
        return None