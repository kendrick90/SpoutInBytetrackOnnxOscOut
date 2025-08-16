#!/usr/bin/env python
"""
Download ByteTrack ONNX models from GitHub
Automatically fetches available models from the ByteTrack releases
"""

import os
import json
import urllib.request
import urllib.error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_github_releases(owner="FoundationVision", repo="ByteTrack"):
    """Fetch release information from GitHub API"""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    
    try:
        logger.info(f"Fetching releases from {owner}/{repo}...")
        with urllib.request.urlopen(api_url) as response:
            releases = json.loads(response.read().decode())
        return releases
    except urllib.error.HTTPError as e:
        logger.error(f"Failed to fetch releases: {e}")
        return None

def find_onnx_models(releases):
    """Extract ONNX model URLs from releases"""
    models = {}
    
    for release in releases:
        for asset in release.get('assets', []):
            name = asset['name']
            if name.endswith('.onnx'):
                models[name] = {
                    'url': asset['browser_download_url'],
                    'size': asset['size'],
                    'release': release['tag_name'],
                    'updated': asset['updated_at']
                }
    
    return models

def download_file(url, destination):
    """Download a file with progress reporting"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownload progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
    
    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {destination}")
    urllib.request.urlretrieve(url, destination, reporthook=download_progress)
    print()  # New line after progress
    logger.info("Download complete!")

def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def main():
    # Try to fetch models from GitHub
    releases = get_github_releases()
    
    if not releases:
        print("\nCould not fetch models from GitHub automatically.")
        print("You can manually download ByteTrack models from:")
        print("https://github.com/ifzhang/ByteTrack/releases")
        return
    
    models = find_onnx_models(releases)
    
    if not models:
        print("\nNo ONNX models found in ByteTrack releases.")
        print("You may need to convert PyTorch models to ONNX.")
        print("See: https://github.com/ifzhang/ByteTrack/tree/main/tools")
        return
    
    # Display available models
    print("\n" + "="*60)
    print("Available ByteTrack ONNX Models:")
    print("="*60)
    
    model_list = list(models.items())
    for i, (name, info) in enumerate(model_list, 1):
        print(f"\n{i}. {name}")
        print(f"   Size: {format_size(info['size'])}")
        print(f"   Release: {info['release']}")
        print(f"   Updated: {info['updated'][:10]}")
        
        # Try to guess input size from name
        if '384x640' in name or '384_640' in name:
            print(f"   Input Size: 384x640 (likely)")
        elif '608x1088' in name or '608_1088' in name:
            print(f"   Input Size: 608x1088 (likely)")
        elif '800x1440' in name or '800_1440' in name:
            print(f"   Input Size: 800x1440 (likely)")
        elif '_s_' in name or 'bytetrack_s' in name.lower():
            print(f"   Input Size: Small variant (likely 384x640)")
        elif '_m_' in name or 'bytetrack_m' in name.lower():
            print(f"   Input Size: Medium variant (likely 608x1088)")
        elif '_l_' in name or 'bytetrack_l' in name.lower():
            print(f"   Input Size: Large variant (likely 800x1440)")
        elif '_x_' in name or 'bytetrack_x' in name.lower():
            print(f"   Input Size: Extra large variant")
    
    print("\n" + "="*60)
    
    # Get user choice
    try:
        choice = input(f"\nWhich model would you like to download? (1-{len(model_list)}, or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            print("Download cancelled")
            return
        
        idx = int(choice) - 1
        if idx < 0 or idx >= len(model_list):
            print("Invalid choice")
            return
            
    except ValueError:
        print("Invalid input")
        return
    
    # Download selected model
    model_name, model_info = model_list[idx]
    
    model_dir = "byte_tracker/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    destination = os.path.join(model_dir, model_name)
    
    if os.path.exists(destination):
        overwrite = input(f"\n{model_name} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Download cancelled")
            return
    
    try:
        download_file(model_info['url'], destination)
        print(f"\nModel downloaded successfully to: {destination}")
        print(f"\nTo use this model, run:")
        print(f"python SpoutByteTrackOnnxOsc.py --model {destination}")
        
        # Suggest input size based on model name
        if '384' in model_name and '640' in model_name:
            print(f"\nSuggested command:")
            print(f"python SpoutByteTrackOnnxOsc.py --model {destination} --input_shape 384,640")
        elif '608' in model_name and '1088' in model_name:
            print(f"\nSuggested command:")
            print(f"python SpoutByteTrackOnnxOsc.py --model {destination} --input_shape 608,1088")
            
    except Exception as e:
        logger.error(f"Failed to download: {e}")

if __name__ == "__main__":
    main()