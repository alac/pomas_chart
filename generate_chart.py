import os
from PIL import Image, ImageDraw, ImageFont
from fonts.ttf import Roboto
from typing import Optional
import imagehash

FACE_SCALE = 1.2
NORMALIZED_FACE_SIZE = (int(53 * FACE_SCALE), int(33 * FACE_SCALE))
COUNT_DIMENSIONS = (100, 50)


class FacesFolder:
    def __init__(self, folder_path: str, image_path: str, count: int):
        self.folder_path = folder_path
        self.image_path = image_path
        self.count = count
        self._phash = None  # type: Optional[str]

    def phash(self) -> str:
        if not self._phash:
            file_path = os.path.join(self.folder_path, self.image_path)
            self._phash = str(imagehash.phash(Image.open(file_path), hash_size=6))
        return self._phash


def generate_chart_from_grouped_faces_folder(grouped_faces_folder: str, chart_filename: str):
    """
    Takes a 'grouped faces folder' and generates a chart from it.
    :param grouped_faces_folder:
    :param chart_filename:
    :return:
    """
    faces_folders_by_count = {}
    for sub_folder in os.listdir(grouped_faces_folder):
        sub_path = os.path.join(grouped_faces_folder, sub_folder)
        if os.path.isdir(sub_path):
            all_images = [f for f in os.listdir(sub_path) if f.endswith(".jpg")]
            count_bucket = faces_folders_by_count.get(len(all_images), [])
            count_bucket.append(FacesFolder(sub_path, all_images[0], len(all_images)))
            faces_folders_by_count[len(all_images)] = count_bucket
    images_in_row = 10
    layout = generate_layout(faces_folders_by_count, images_in_row)
    make_chart(chart_filename, layout, images_in_row)


def generate_layout(faces_folders_by_count: dict[int, list[FacesFolder]], images_in_row: int)\
        -> list[tuple[int, list[list[str]]]]:
    """
    Takes the frequency dictionary and generates a 'layout' for the face images.
    :param faces_folders_by_count:
    :param images_in_row:
    :return: layout: a list where each element is a tuple of a (COUNT, CONTENT_OF_ROWS)
        the COUNT is the label for the row
        the CONTENT_OF_ROWS is a list where each element describes a ROW
        a ROW is a list[str], where each str is the path of an image
    """
    all_counts = sorted([count for count in faces_folders_by_count])

    layout = []
    for count in reversed(all_counts):
        rows_for_count = []
        all_faces_for_count = faces_folders_by_count[count]
        sorted(all_faces_for_count, key=lambda x: x.phash())
        while all_faces_for_count:
            current_row = []
            for face_folder in all_faces_for_count[:images_in_row]:
                current_row.append(os.path.join(face_folder.folder_path, face_folder.image_path))
            rows_for_count.append(current_row)
            all_faces_for_count = all_faces_for_count[images_in_row:]
        layout.append((count, rows_for_count))
    return layout


def make_chart(filename: str, layout: list[tuple[int, list[list[str]]]], images_in_row: int):
    """
    arranges the chart elements on an image canvas and writes them to a file.
    :param filename:
    :param layout: a list where each element is a tuple of a (COUNT, CONTENT_OF_ROWS)
        the COUNT is the label for the row
        the CONTENT_OF_ROWS is a list where each element describes a ROW
        a ROW is a list[str], where each str is the path of an image
    :param images_in_row:
    :return:
    """
    total_width = COUNT_DIMENSIONS[0] + images_in_row*NORMALIZED_FACE_SIZE[0]
    total_height = 0
    for _, image_rows in layout:
        total_height += max(COUNT_DIMENSIONS[1], len(image_rows) * NORMALIZED_FACE_SIZE[1])

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
    generate_chart_from_grouped_faces_folder(r"out\group", r"out\chart.png")
