import ffmpeg
import os

# Directory containing the images
image_dir = 'img'
# Output video file name
output_file = 'bayes_filter.mp4'

# Ensure the output file does not already exist
if os.path.exists(output_file):
    os.remove(output_file)

# Use ffmpeg to convert the images to a video
(
    ffmpeg
    .input(os.path.join(image_dir, 'step%01d.png'), framerate=2)
    .output(output_file)
    .run()
)

print(f"Video has been saved as {output_file}")