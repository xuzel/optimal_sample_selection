from PIL import Image

def resize_image(input_image, output_image, new_size):
    image = Image.open(input_image)
    resized_image = image.resize(new_size)
    resized_image.save(output_image)

# Example usage
input_image = r"D:\文档\人工智能\example\Optimal-sample-selection\static\art1.png"
output_image = r"D:\文档\人工智能\example\Optimal-sample-selection\static\art1-resized.png"
new_size = (2000, 1440)

resize_image(input_image, output_image, new_size)