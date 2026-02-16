import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Setup class names
class_names = [
    'آ','ا','ب','پ', 'ت','ث','ج','چ','ح','خ',
    'د','ذ','ر','ز','ژ','س','ش','ص','ض','ط',
    'ظ','ع','غ','ف','ق','ک','گ','ل','لا','م',
    'ن','و','ء','ها','ی'
]

# Path to your image directory
image_dir = "examples/*.png"  # Replace with your directory path and file extension

# Load images
images = []
for img_path in glob.glob(image_dir):
    images.append(mpimg.imread(img_path))

# Display images with labels (e.g., file names)
plt.figure(figsize=(9, 4))
columns = 8  # Number of columns in the grid
for i, image in enumerate(images):
    plt.subplot(len(images) // columns + 1, columns, i + 1)
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    # Optional: Add label (e.g., filename)
    plt.title(f"Image {class_names[i+1]}", fontsize=8)

# plt.tight_layout()
plt.show()   