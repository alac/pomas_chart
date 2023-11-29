import os
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto

FACE_SCALE = 1.0
NORMALIZED_FACE_SIZE = (int(53 * FACE_SCALE), int(33 * FACE_SCALE))
COUNT_DIMENSIONS = (100, 50)


class FacesFolder:
    def __init__(self, folder_path: str, image_path: str, count: int):
        self.folder_path = folder_path
        self.image_path = image_path
        self.count = count


def generate_chart_from_grouped_faces_folder(grouped_faces_folder: str, chart_filename: str):
    faces_folders_by_count = {}
    for sub_folder in os.listdir(grouped_faces_folder):
        sub_path = os.path.join(grouped_faces_folder, sub_folder)
        if os.path.isdir(sub_path):
            all_images = [f for f in os.listdir(sub_path) if f.endswith(".jpg")]
            count_bucket = faces_folders_by_count.get(len(all_images), [])
            count_bucket.append(FacesFolder(sub_path, all_images[0], len(all_images)))
            faces_folders_by_count[len(all_images)] = count_bucket
    layout, width, height = generate_layout(faces_folders_by_count)
    make_chart(chart_filename, layout, width, height)


def generate_layout(faces_folders_by_count: dict[int, list[FacesFolder]]) -> (list[tuple[int, list[str]]], int, int):
    images_in_row = 10
    all_counts = sorted([count for count in faces_folders_by_count])
    total_width = COUNT_DIMENSIONS[0] + images_in_row*NORMALIZED_FACE_SIZE[0]
    total_height = 0

    layout = []
    for count in reversed(all_counts):
        rows_for_count = []
        all_faces_for_count = faces_folders_by_count[count]
        while all_faces_for_count:
            current_row = []
            for face_folder in all_faces_for_count[:10]:
                current_row.append(os.path.join(face_folder.folder_path, face_folder.image_path))
            rows_for_count.append(current_row)
            all_faces_for_count = all_faces_for_count[10:]
        layout.append((count, rows_for_count))
        total_height += max(COUNT_DIMENSIONS[1], len(rows_for_count)*NORMALIZED_FACE_SIZE[1])
    return layout, total_width, total_height


def make_chart(filename: str, layout: list[tuple[int, list[str]]], total_width: int, total_height: int):
    chart = Image.new("RGB", (total_width, total_height), "white")
    chart_draw = ImageDraw.Draw(chart)

    y = 0
    for count, image_rows in layout:
        label_text = str(count)
        font_size = fit_text_to_height(label_text, COUNT_DIMENSIONS[1])
        chart_draw.text((0, y), label_text, (0, 0, 0), get_font(font_size))

        images_start_x = COUNT_DIMENSIONS[0]
        image_y = y
        for image_row in image_rows:
            image_x = images_start_x
            for ip in image_row:
                with Image.open(ip) as i:
                    chart.paste(i, (image_x, image_y))
                    image_x += NORMALIZED_FACE_SIZE[0]
            image_y += NORMALIZED_FACE_SIZE[1]
        y = max(y + COUNT_DIMENSIONS[1], image_y)
    chart.save(filename)


def compute_height(fontsize, text):
    img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(img)
    _x, _y, width, height = draw.textbbox((0, 0), text, font=get_font(fontsize))
    return height


def get_font(fontsize):
    return ImageFont.truetype(Roboto, fontsize)


def fit_text_to_height(text, height):
    # print(f"fit_text_to_height {text}, {height}")
    max_size = 1
    result = compute_height(max_size, text)
    while result < height:
        max_size *= 2
        result = compute_height(max_size, text)
    min_size = max_size // 2
    safe_size = min_size
    for x in range(10):
        test_size = int(min_size + (x / 10.0) * (max_size - min_size))
        if compute_height(max_size, text) < height:
            safe_size = test_size
        else:
            return safe_size


if __name__ == "__main__":
    generate_chart_from_grouped_faces_folder("out\group", "chart.jpg")