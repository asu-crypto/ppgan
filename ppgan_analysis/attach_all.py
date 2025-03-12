from PIL import Image

# Load your 20 images (replace with your image file paths)

image_paths = []#"image1.jpg", "image2.jpg", "image3.jpg", ...]  # Provide paths to all 20 images
for i in range(20):
    image_paths.append(f"exp/inverted_deep_convolution_0_secure_celeba_bsr_1_bsf_1_{i}.jpg")
# Ensure all images have the same dimensions (you may need to resize them)
image_list = [Image.open(image_path) for image_path in image_paths]

# Assuming all images have the same width and height
width, height = image_list[0].size

# Define the spacing size in pixels
spacing = 2  # Adjust this value as needed

# Calculate the size of the grid image
grid_width = width * 10 + spacing * 9
grid_height = height * 2 + spacing

# Create a new blank image
grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

# Paste the 20 images onto the grid image with spacing
for i in range(20):
    x = (i % 10) * (width + spacing)
    y = (i // 10) * (height + spacing)
    grid_image.paste(image_list[i], (x, y))

# Save the resulting image
grid_image.save("grid_image_2x10_fl.jpg")