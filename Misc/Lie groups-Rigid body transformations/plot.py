import ffmpeg
import os

# Directory containing the images
image_dir = 'animation'
# Output video file name
output_file = 'lie_group_based_3d_animation.mp4'

# Ensure the output file does not already exist
if os.path.exists(output_file):
    os.remove(output_file)

# Use ffmpeg to convert the images to a video
(
    ffmpeg
    .input(os.path.join(image_dir, 'rotation_translation_%03d.png'), framerate=10)
    .output(output_file)
    .run()
)

print(f"Video has been saved as {output_file}")