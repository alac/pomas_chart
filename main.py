import os
import cv2
import numpy as np
import imagehash
import tqdm
from PIL import Image
import json
import shutil

from detect_image import find_location_cv_multi, write_debug_image
from generate_chart import NORMALIZED_FACE_SIZE, generate_chart_from_grouped_faces_folder, FACE_SCALE

IMAGE_ICON_I = r"examples/crops/icon_i.jpg"
IMAGE_LOGO = r"examples/crops/logo.jpg"
IMAGE_BADGE = r"examples/crops/badge_check.jpg"

IMAGE_BOTTOM_EDGE = r"examples/crops/bottom_edge.jpg"
IMAGE_TOP_CORNER = r"examples/crops/top_corner.jpg"
CROPPED_IMAGE_MATCH_THRESHOLD = .70

SCALING_CACHE = {}
SCALING_CACHE_FILE = "scaling_cache.json"


class CroppedResultsException(ValueError):
    pass


def extract_and_group_faces_from_folder(folder: str):
    # reset the output folder
    shutil.rmtree("out")
    os.makedirs("out", exist_ok=True)

    if os.path.exists(SCALING_CACHE_FILE):
        with open(SCALING_CACHE_FILE, encoding="utf-8") as f:
            global SCALING_CACHE
            SCALING_CACHE = json.load(f)

    face_groups = {}
    entries = os.scandir(folder)
    file_paths = [entry.path for entry in entries]
    sorted_file_paths = sorted(file_paths, key=os.path.getsize, reverse=True)
    for fp in tqdm.tqdm(sorted_file_paths, "scanning files"):
        if os.path.isfile(fp):
            try:
                faces = search_for_faces(fp)
                add_faces_to_groups(face_groups, faces)
            except Exception as e:
                print("Error while processing ", fp)
                raise e

    all_faces_folder = os.path.join("out", "all_faces")
    os.makedirs(all_faces_folder, exist_ok=True)
    for group_name in face_groups:
        group_folder = os.path.join("out", "group", group_name)
        os.makedirs(group_folder, exist_ok=True)
        for i, face_image in enumerate(face_groups[group_name]):
            out_path = os.path.join("out", "group", group_name, f"{group_name}_{i}.jpg")
            cv2.imwrite(out_path, face_image)
            out_path = os.path.join("out", "all_faces", f"{group_name}_{i}.jpg")
            cv2.imwrite(out_path, face_image)

    print("SCALING_CACHE", SCALING_CACHE)


def search_for_scaling(chart_image_path: str, chart_img: cv2.typing.MatLike, reference_image: str) -> (float, float):
    global SCALING_CACHE
    scaling_cache_key = f"{chart_image_path}:{reference_image}"
    if scaling_cache_key in SCALING_CACHE:
        best_scale, best_match = SCALING_CACHE[scaling_cache_key]
        return best_scale, best_match

    icon_img = cv2.imread(reference_image)
    original_height, original_width, _ = icon_img.shape

    best_match = 0
    best_scale = 1.0
    for width in tqdm.tqdm(range(int(original_width * 1.2), 10, -1), f"searching for scaling"):
        height = int(original_height * (width / original_width))
        resized_icon = cv2.resize(icon_img, (width, height), interpolation=cv2.INTER_AREA)
        matches = find_location_cv_multi(resized_icon, chart_img, .5, max_count=1)
        if len(matches) > 0:
            strength, _box = matches[0]
            if strength > best_match:
                best_match = strength
                best_scale = width/original_width
            if best_match > .9 and strength < best_match - .05:
                break
    print(f"best match found: {best_scale}, accuracy {best_match} for {chart_image_path}")

    SCALING_CACHE[scaling_cache_key] = best_scale, best_match
    with open(SCALING_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(SCALING_CACHE, f)

    return best_scale, best_match


def search_for_faces(chart_image_path: str) -> list[cv2.typing.MatLike]:
    chart_img = cv2.imread(chart_image_path)
    if chart_img is None:
        print("error reading chart:", chart_image_path)
        return []

    # scale down the image since it's the difference between taking 1 minute per image and 2 seconds.
    original_height, original_width, _ = chart_img.shape
    if max(original_height, original_width) > 2048:
        scale = min(2048/original_height, 2048/original_width)
        width = int(original_width * scale)
        height = int(original_height * scale)
        chart_img = cv2.resize(chart_img, (width, height), interpolation=cv2.INTER_AREA)

    # we're using OpenCV to guess the locations. determine the ratio of our template image to the chart image.
    best_scale, best_match = 0, 0
    for scaling_ref in [IMAGE_TOP_CORNER, IMAGE_LOGO, IMAGE_ICON_I, IMAGE_BADGE]:
        test_scale, test_match = search_for_scaling(chart_image_path, chart_img, scaling_ref)
        if test_match > .94:
            best_scale, best_match = test_scale, test_match
            break
        if test_match > best_match:
            best_scale, best_match = test_scale, test_match
    scale = best_scale

    corner_img = cv2.imread(IMAGE_TOP_CORNER)
    original_height, original_width, _ = corner_img.shape
    width = int(original_width * scale)
    height = int(original_height * scale)
    resized_corner = cv2.resize(corner_img, (width, height), interpolation=cv2.INTER_AREA)
    matches = find_location_cv_multi(resized_corner, chart_img, .8, max_count=6)

    face_matches = []
    for strength, bounding_box in matches:
        try:
            face_locations = face_locations_near_match(chart_img, scale, bounding_box)
            [face_matches.append((strength, face_location)) for face_location in face_locations]
        except CroppedResultsException:
            print(f"Image had cropped results: ", chart_image_path)

    just_faces_images = []
    for _strength, face_location in face_matches:
        x, y, w, h = face_location
        just_faces_images.append(chart_img[y:y + h, x:x + w])

    threshold = .95
    while len(just_faces_images) > 15 and threshold > .7:
        # print("trimming")
        just_faces_images = deduplicate_faces(just_faces_images, threshold)
        threshold -= .05

    _directory, filename = os.path.split(chart_image_path)
    if len(just_faces_images) != 15:
        print(f"image count {len(just_faces_images)} for {filename}")

    resized_faces = []
    for i, face_image in enumerate(just_faces_images):
        face_image = cv2.resize(face_image, NORMALIZED_FACE_SIZE, interpolation=cv2.INTER_AREA)
        resized_faces.append(face_image)

    all_faces_folder = os.path.join("out", "outlined_charts")
    os.makedirs(all_faces_folder, exist_ok=True)
    out_path = os.path.join("out", "outlined_charts", filename)
    write_debug_image(chart_img, face_matches, out_path)

    return resized_faces


def face_locations_near_match(chart_img: cv2.typing.MatLike, scale: float, matched_corner_box: tuple[int, int, int, int]
                              ) -> list[tuple[int, int, int, int]]:
    """
    Takes a chart and the location of the corner of a "battle result".
    Returns the location of trainer faces for that "battle result".
    :param chart_img:
    :param scale: the estimated amount that the "corner" marker needs to be scaled to fit the chart image
    :param matched_corner_box:
    :return:
    """
    x, y, _width, _height = matched_corner_box

    slot1 = (64-3, 65-3, 182, 116+10)
    slot2 = (280-6, 65-3, 182, 116+10)
    slot3 = (496-9, 65-3, 182, 116+10)

    margin_color = np.array([221, 194, 80])
    bottom_shadow = np.array([180, 165, 92])

    absolute_bounding_boxes = []
    for sx, sy, sw, sh in [slot1, slot2, slot3]:
        # the template is the right size, but needs to be shifted on wider screens
        # easier to shift pixel by pixel than to make the templates dynamic (?)
        i = 0
        for i in range(10):
            dist1 = compute_color_dist_column(
                chart_img,
                margin_color,
                int(x + scale * sx) + i,
                int(y + scale * sy),
                int(y + scale * sy) + int(scale * sh))
            dist2 = compute_color_dist_column(
                chart_img,
                margin_color,
                int(x + scale * sx) + i + int(scale * sw),
                int(y + scale * sy),
                int(y + scale * sy) + int(scale * sh))
            # print(f"dist1 {dist1} dist2 {dist2}")
            if dist1 >= dist2:
                break
            if 45 <= dist1:
                break
        # print(f"padding by {i}")

        # nudge the template down to meet the bottom edge of the trainer's portrait.
        j = 0
        for j in range(22):
            dist2 = compute_color_dist_row(
                chart_img,
                bottom_shadow,
                int(x + scale * sx),
                int(x + scale * sx) + int(scale * sw/2),
                int(y + scale * sy + scale * sh) + j)
            # print(f"dist1 {dist1} dist2 {dist2}")
            # 45 was experimentally determined (30 breaks low res image, 50 breaks high-res)
            if 45 >= dist2:
                break

        absolute_bounding_boxes.append((
            int(x + scale * sx) + i,
            int(y + scale * sy) + j,
            int(scale * sw),
            int(scale * sh),
        ))
    return absolute_bounding_boxes


def compute_color_dist_column(image: cv2.typing.MatLike, color, x: int, y_start: int, y_end: int) -> float:
    """
    For a vertical line in the image, compute the median "distance between pixel color and a reference color"
    :param image:
    :param color:
    :param x:
    :param y_start:
    :param y_end:
    :return:
    """
    distances = []
    for y in range(y_start, y_end):
        max_y, max_x, _ = image.shape
        if y < max_y and x < max_x:
            left_color = image[y, x]
            distances.append(np.linalg.norm(color - left_color))
        else:
            raise CroppedResultsException()
    return sorted(distances)[int(len(distances)/2)]


def compute_color_dist_row(image: cv2.typing.MatLike, color, x_start: int, x_end, y) -> float:
    """
    For a horizontal line in the image, compute the median "distance between pixel color and a reference color"
    :param image:
    :param color:
    :param x_start:
    :param x_end:
    :param y:
    :return:
    """
    distances = []
    for x in range(x_start, x_end):
        max_y, max_x, _ = image.shape
        if y < max_y and x < max_x:
            left_color = image[y, x]
            distances.append(np.linalg.norm(color - left_color))
        else:
            raise CroppedResultsException()
    return sorted(distances)[int(len(distances)/2)]


def deduplicate_faces(face_images: list[cv2.typing.MatLike], threshold: float) -> list[cv2.typing.MatLike]:
    de_duped_images = []
    for face_image in face_images:
        has_match = False
        for reference_faces in de_duped_images:
            matches = find_location_cv_multi(face_image, reference_faces, threshold, max_count=1)
            if len(matches):
                has_match = True
        if not has_match:
            de_duped_images.append(face_image)
    return de_duped_images


def add_faces_to_groups(face_groups: dict, new_faces: list[cv2.typing.MatLike]):
    """
    Cluster matching faces.
    :param face_groups:
    :param new_faces:
    :return:
    """
    for face_image in new_faces:
        best_group = None
        best_accuracy = 0
        for group_key in tqdm.tqdm(face_groups, f"grouping faces", disable=True):
            possible_group = face_groups[group_key]
            best_trainer_accuracy = 0
            best_pokemon_accuracy = 0
            for group_member in possible_group:
                trainer_accuracy, pokemon_accuracy = match_face_to_reference(face_image, group_member,
                                                                             CROPPED_IMAGE_MATCH_THRESHOLD)
                best_trainer_accuracy = max(trainer_accuracy, best_trainer_accuracy)
                best_pokemon_accuracy = max(pokemon_accuracy, best_pokemon_accuracy)
                trainer_accuracy, pokemon_accuracy = match_face_to_reference(group_member, face_image,
                                                                             CROPPED_IMAGE_MATCH_THRESHOLD)
                best_trainer_accuracy = max(trainer_accuracy, best_trainer_accuracy)
                best_pokemon_accuracy = max(pokemon_accuracy, best_pokemon_accuracy)
            if min(best_pokemon_accuracy, best_trainer_accuracy) > best_accuracy:
                best_accuracy = min(best_pokemon_accuracy, best_trainer_accuracy)
                best_group = possible_group
        if best_group:
            best_group.append(face_image)
        else:
            phash = imagehash.phash(Image.fromarray(face_image), hash_size=6)
            face_groups[str(phash)] = [face_image]


def match_face_to_reference(face_image: cv2.typing.MatLike, reference_image: cv2.typing.MatLike,
                            threshold: float) -> (float, float):
    image_1_width, image_1_height, _ = face_image.shape
    cropped_face = face_image[int(image_1_height*.2):int(image_1_height*.9),
                              int(image_1_width*.1):int(image_1_width*.6)]
    matches = find_location_cv_multi(reference_image, cropped_face, threshold, max_count=1)
    if len(matches) == 0:
        return 0, 0
    trainer_accuracy, _ = matches[0]

    pokemon_accuracy = 0
    measured_bounding_box = (int(33 * FACE_SCALE), int(16 * FACE_SCALE), int(19 * FACE_SCALE), int(16 * FACE_SCALE))
    pkm_x, pkm_y, pkm_w, pkm_h = measured_bounding_box
    cropped_pokemon = face_image[pkm_y: pkm_y+pkm_h, pkm_x: pkm_x+pkm_w]
    matches = find_location_cv_multi(reference_image, cropped_pokemon, threshold, max_count=1)
    if len(matches) != 0:
        pokemon_accuracy, _ = matches[0]

    return trainer_accuracy, pokemon_accuracy


if __name__ == '__main__':
    extract_and_group_faces_from_folder(r"in")
    generate_chart_from_grouped_faces_folder(r"out\group", r"out\chart.jpg")
