from PIL import Image

# Load your four images
image1 = Image.open("exp/real_fl_deep_convolution_celeba_1_1.jpg")
image2 = Image.open("exp/real_fl_deep_convolution_celeba_1_2.jpg")
image3 = Image.open("exp/real_fl_deep_convolution_celeba_1_3.jpg")
image4 = Image.open("exp/real_fl_deep_convolution_celeba_1_4.jpg")
# image1 = Image.open("exp/inverted_deep_convolution_0_secure_celeba_bsr_1_bsf_1_15.jpg")
# image2 = Image.open("exp/inverted_deep_convolution_1_secure_celeba_bsr_1_bsf_32_15.jpg")
# image3 = Image.open("exp/inverted_deep_convolution_2_secure_celeba_bsr_1_bsf_32_15.jpg")
# image4 = Image.open("exp/inverted_deep_convolution_3_secure_celeba_bsr_1_bsf_32_15.jpg")

# Ensure all images have the same dimensions (you may need to resize them)
width, height = image1.size
image2 = image2.resize((width, height))
image3 = image3.resize((width, height))
image4 = image4.resize((width, height))

# Define the spacing size in pixels
spacing = 5  # Adjust this value as needed

# Create a new blank image
grid_width = width * 2 + spacing
grid_height = height * 2 + spacing
grid_image = Image.new("RGB", (grid_width, grid_height), color="black")

# Paste the four images onto the grid image with spacing
grid_image.paste(image1, (0, 0))
grid_image.paste(image2, (width + spacing, 0))
grid_image.paste(image3, (0, height + spacing))
grid_image.paste(image4, (width + spacing, height + spacing))

# Save the resulting image
grid_image.save("grid_image_real.jpg")