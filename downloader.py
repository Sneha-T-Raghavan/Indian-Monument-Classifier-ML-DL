from bing_image_downloader import downloader    

# Specify the search term and download directory
search_term = "Brihadeshwara Temple"
output_dir = r"C:/Amrita/Sem 5/ML/CNN/Webscrapped_Data/training data"

# Download 100 images
downloader.download(search_term, limit=100, output_dir=output_dir, adult_filter_off=True, force_replace=False, timeout=60)

print(f"Images of {search_term} have been downloaded into the folder '{output_dir}'.")
