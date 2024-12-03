import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Directory to save images
SAVE_DIR = "ikea_images"

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
    # "Sofasets": "https://www.ikea.com/us/en/cat/sofas-fu003/",
    # "Beds": "https://www.ikea.com/us/en/cat/beds-bm003/",
    # "Chairs": "https://www.ikea.com/us/en/cat/chairs-fu002/",
    # "Shelves": "https://www.ikea.com/us/en/cat/bookcases-shelving-units-st002/",
    # "TableDesk": "https://www.ikea.com/us/en/cat/tables-desks-fu004/",
    # "Wardrobes": "https://www.ikea.com/us/en/cat/armoires-wardrobes-19053/",
    # "Dining Sets": "https://www.ikea.com/us/en/cat/dining-sets-19145/",
    # "Vases": "https://www.ikea.com/us/en/cat/vases-10776/",
    # "Frames": "https://www.ikea.com/us/en/cat/wall-art-10788/",
    # "FloorLamp": "https://www.ikea.com/us/en/cat/floor-lamps-10731/",
}

# Scrape each category
for category, url in categories.items():
    print(f"Scraping {category}...")
    scrape_category(url, category)
