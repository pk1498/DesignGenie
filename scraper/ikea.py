import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Directory to save images
SAVE_DIR = "ikea_images_new"

# Function to download an image
def download_image(url, folder, filename):
    full_folder_path = os.path.join(SAVE_DIR, folder)
    os.makedirs(full_folder_path, exist_ok=True)
    
    filepath = os.path.join(full_folder_path, filename)  # Full path to save the image
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filepath}")
    else:
        print(f"Failed to download: {url}")


# Function to scrape images from a category page
def scrape_category(url, folder):
    print(folder)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find all image tags
    images = soup.find_all("img")
    for idx, img in enumerate(images):
        # Extract image source
        img_url = img.get("src") or img.get("data-src")
        if img_url:
            img_url = urljoin(url, img_url)  # Make absolute URL
            download_image(img_url, folder, f"{folder}_{idx}.jpg")

# IKEA URLs for categories (replace these with actual IKEA category URLs)
categories = {
    # "Sofa": "https://www.ikea.com/us/en/cat/sofas-fu003/",
    # "Bed": "https://www.ikea.com/us/en/cat/beds-bm003/",
    # "Chair": "https://www.ikea.com/us/en/cat/chairs-fu002/",
    # "Desk": "https://www.ikea.com/us/en/cat/desks-for-home-20651/",
    # "Table": "https://www.ikea.com/us/en/cat/accent-tables-10705/?page=2",
    # "Wardrobe": "https://www.ikea.com/us/en/cat/armoires-wardrobes-19053/",
    # "Dining Set": "https://www.ikea.com/us/en/cat/dining-sets-19145/",
    # "Vase": "https://www.ikea.com/us/en/cat/vases-10776/",
    # "Photo Frame": "https://www.ikea.com/us/en/cat/wall-art-10788/",
    # "Floor lamp": "https://www.ikea.com/us/en/cat/floor-lamps-10731/",
    # "Kitchen shelf": "https://www.ikea.com/us/en/cat/kitchen-cabinets-700292/?page=2",
    # "Bathroom shelf": "https://www.ikea.com/us/en/cat/bathroom-vanities-20719/"
}

# Scrape each category
for category, url in categories.items():
    print(f"Scraping {category}...")
    scrape_category(url, category)





# ------------- Run below code for getting annotated dataset from roboflow --------------------

# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="nJ8wpptmMI7nb0VkwOBX")
# project = rf.workspace("project-1iawg").project("designgenie")
# version = project.version(1)
# dataset = version.download("yolov5")
                
