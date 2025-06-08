import os

folder_path = './val/label/'
output_path = './val/list/'
output_dir = os.path.dirname(output_path)
output_file = output_path + 'val.txt'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = os.listdir(folder_path)


image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]


with open(output_file, 'w') as f:
    for image in sorted(image_files):
        f.write(f"{image}\n")

print(f"Information has been reported in {output_path}")