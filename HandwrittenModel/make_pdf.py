from PIL import Image
import os


def generate_pdf():
    images_names = []
    for file in os.listdir('HandwrittenModel/generated/'):
        images_names.append(file)

    images_names.sort()

    images = [Image.open(
        f'HandwrittenModel/generated/{img}') for img in images_names]

    images = [img.convert('L') for img in images]

    images = [img.resize((800, 80)) for img in images]

    new_img_size = (900, 1000)

    img_count = 0
    new_image = Image.new('RGB', new_img_size, (255, 255, 255))
    current_y = 50

    new_images_list = []

    for img in images:
        if img_count == 10:
            new_images_list.append(new_image)
            new_image = Image.new('RGB', new_img_size, (255, 255, 255))
            img_count = 0
            current_y = 50
        new_image.paste(img, (50, current_y))
        img_count += 1
        current_y += img.size[1]-10

    if img_count > 0:
        new_images_list.append(new_image)

    img1 = new_images_list[0]
    new_images_list.remove(new_images_list[0])
    img1.save('HandwrittenModel/Handwritten.pdf',
              save_all=True, append_images=new_images_list)

    for name in images_names:
        os.remove("HandwrittenModel/generated/" + name)
