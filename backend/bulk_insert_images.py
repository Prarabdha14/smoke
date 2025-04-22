import os
from database.db import save_result

# Base path to your image folders
base_path = "images"
image_types = ["original", "preprocessed", "cyclegan", "srgan"]

def bulk_insert_images():
    for img_type in image_types:
        folder = os.path.join(base_path, img_type)
        if not os.path.exists(folder):
            print(f"Folder '{folder}' does not exist. Skipping...")
            continue

        # Iterate over all images in the folder
        for filename in os.listdir(folder):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(folder, filename)
                
                # Open the image as bytes
                with open(filepath, "rb") as f:
                    image_bytes = f.read()

                # Save to the database
                save_result(
                    filename=filename,
                    image_bytes=image_bytes,
                    prediction="N/A",  # Add your model's prediction here
                    image_type=img_type  # Image type (original, cyclegan, etc.)
                )
                print(f"Saved {filename} to the database.")

    print("All images have been inserted successfully.")

# Run the bulk insert function
if __name__ == "__main__":
    bulk_insert_images()
