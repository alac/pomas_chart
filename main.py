import os.path
import cv2
import numpy as np
import imagehash
import tqdm
from PIL import Image

from detect_image import find_location_cv_multi, write_debug_image

IMAGE_ICON_I = r"examples/crops/icon_i.jpg"
IMAGE_LOGO = r"examples/crops/logo.jpg"
IMAGE_BOTTOM_EDGE = r"examples/crops/bottom_edge.jpg"
IMAGE_TOP_CORNER = r"examples/crops/top_corner.jpg"
DEDUPE_HASH_SIZE = 5
NORMALIZED_FACE_SIZE = (2*53, 2*33)


def process_folder(folder: str):
    face_groups = {}
    for file in tqdm.tqdm(os.listdir(folder), "scanning files"):
        fp = os.path.join(folder, file)
        if os.path.isfile(fp):
            faces = search_for_faces(fp)
            add_faces_to_groups(face_groups, faces)

    for group_name in face_groups:
        out_folder = os.path.join("out", "group", group_name)
        os.makedirs(out_folder, exist_ok=True)
        for i, face_image in enumerate(face_groups[group_name]):
            # out_path = os.path.join("out", "group", group_name, f"{group_name}_{i}.jpg")
            # cv2.imwrite(out_path, face_image)
            out_path = os.path.join("out", "group", f"{group_name}_{i}.jpg")
            cv2.imwrite(out_path, face_image)


def search_for_scaling(chart_image_path: str, chart_img: cv2.typing.MatLike, reference_image: str) -> (float, float):
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
    print(f"best match found: {best_scale}, accuracy {best_match} for {chart_image_path}")
    return best_scale, best_match


def search_for_faces(chart_image_path: str) -> list[cv2.typing.MatLike]:
    chart_img = cv2.imread(chart_image_path)
    original_height, original_width, _ = chart_img.shape
    if max(original_height, original_width) > 2048:
        scale = min(2048/original_height, 2048/original_width)
        width = int(original_width * scale)
        height = int(original_height * scale)
        chart_img = cv2.resize(chart_img, (width, height), interpolation=cv2.INTER_AREA)

    scale, strength = search_for_scaling(chart_image_path, chart_img, IMAGE_LOGO)
    if strength < .90:
        print(f"poor match quality {strength}, trying {IMAGE_ICON_I}")
        scale, strength = search_for_scaling(chart_image_path, chart_img, IMAGE_ICON_I)
        if strength < .90:
            print(f"poor match quality {strength}, skipping {chart_image_path}")
            return []

    corner_img = cv2.imread(IMAGE_TOP_CORNER)
    original_height, original_width, _ = corner_img.shape
    width = int(original_width * scale)
    height = int(original_height * scale)
    resized_corner = cv2.resize(corner_img, (width, height), interpolation=cv2.INTER_AREA)

    matches = find_location_cv_multi(resized_corner, chart_img, .8, max_count=6)
    _directory, filename = os.path.split(chart_image_path)
    # out_path = os.path.join("out", "corners_" + filename)
    # write_debug_image(chart_img, matches, out_path)

    face_matches = []
    for strength, bounding_box in matches:
        face_locations = face_locations_near_match(chart_img, scale, bounding_box)
        [face_matches.append((strength, face_location)) for face_location in face_locations]
    # out_path = os.path.join("out", "faces_" + filename)
    # write_debug_image(chart_img, face_matches, out_path)

    just_faces_images = []
    for _strength, face_location in face_matches:
        x, y, w, h = face_location
        just_faces_images.append(chart_img[y:y + h, x:x + w])

    threshold = .95
    while len(just_faces_images) > 15 and threshold > .7:
        # print("trimming")
        just_faces_images = deduplicate_faces(just_faces_images, threshold)
        threshold -= .10

    if len(just_faces_images) != 15:
        print(f"image count {len(just_faces_images)} for {filename}")

    resized_faces = []
    for i, face_image in enumerate(just_faces_images):
        face_image = cv2.resize(face_image, NORMALIZED_FACE_SIZE, interpolation=cv2.INTER_AREA)
        resized_faces.append(face_image)
    #     out_path = os.path.join("out", f"just_faces_{filename}_{i}.jpg")
    #     cv2.imwrite(out_path, face_image)
    return resized_faces


def face_locations_near_match(chart_img: cv2.typing.MatLike, scale: float, matched_corner_box: tuple[int, int, int, int]
                              ) -> list[tuple[int, int, int, int]]:
    x, y, _width, _height = matched_corner_box

    slot1 = (64, 65, 182, 116)
    slot2 = (280, 65, 182, 116)
    slot3 = (496, 65, 182, 116)

    margin_color = np.array([221, 194, 80])

    absolute_bounding_boxes = []
    for sx, sy, sw, sh in [slot1, slot2, slot3]:
        # the template is the right size, but needs to be shifted on wider screens
        # easier to shift pixel by pixel than to make the templates dynamic (?)
        i = 0
        for i in range(10):
            dist1 = compute_color_dist(chart_img, margin_color, int(x + scale * sx) + i, int(y + scale * sy), int(y + scale * sy) + int(scale * sh))
            dist2 = compute_color_dist(chart_img, margin_color, int(x + scale * sx) + i + int(scale * sw), int(y + scale * sy), int(y + scale * sy) + int(scale * sh))
            # print(f"dist1 {dist1} dist2 {dist2}")
            if dist1 >= dist2:
                break
        # print(f"padding by {i}")

        absolute_bounding_boxes.append((
            int(x + scale * sx) + i,
            int(y + scale * sy),
            int(scale * sw),
            int(scale * sh),
        ))
    return absolute_bounding_boxes


def compute_color_dist(image: cv2.typing.MatLike, color, x: int, y_start: int, y_end: int) -> float:
    distances = []
    for y in range(y_start, y_end):
        left_color = image[y, x]
        distances.append(np.linalg.norm(color - left_color))
    return sorted(distances)[int(len(distances)/2)]


def deduplicate_faces(face_images: list[cv2.typing.MatLike], threshold: float) -> list[cv2.typing.MatLike]:
    deduped_images = []
    for face_image in face_images:
        has_match = False
        for reference_faces in deduped_images:
            matches = find_location_cv_multi(face_image, reference_faces, threshold, max_count=1)
            if len(matches):
                has_match = True
        if not has_match:
            deduped_images.append(face_image)
        # else:
        #     print("excluding an image")
    return deduped_images


def add_faces_to_groups(face_groups: dict, new_faces: list[cv2.typing.MatLike]):
    # print(face_groups)
    for face_image in new_faces:
        best_group = None
        best_match = 100.0
        for group_key in tqdm.tqdm(face_groups, f"grouping faces"):
            possible_group = face_groups[group_key]
            # matches = find_location_cv_multi(face_image, possible_group[0], .75, max_count=1)
            # for strength, _box in matches:
            #     if strength > best_match:
            #         best_group = possible_group
            #         best_match = strength
            for group_member in possible_group:
                image_diff = get_image_difference(face_image, group_member)
                if image_diff < .25 and image_diff < best_match:
                    best_match = image_diff
                    best_group = possible_group
        if best_group:
            best_group.append(face_image)
        else:
            phash = imagehash.phash(Image.fromarray(face_image), hash_size=6)
            face_groups[str(phash)] = [face_image]


def get_image_difference(image_1, image_2):
    # per https://stackoverflow.com/a/45485883
    # returns 0 for a perfect match, different images in this context were closer to .9
    first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
    second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

    img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)

    image_1_w, image_1_height, _ = image_1.shape
    x_crop_size = 0
    y_crop_size = 1
    cropped_image_1 = image_1[y_crop_size:image_1_height-2*y_crop_size, x_crop_size:image_1_w-2*x_crop_size]
    img_template_probability_match = cv2.matchTemplate(image_2, cropped_image_1, cv2.TM_CCOEFF_NORMED)[0][0]
    img_template_diff = 1 - img_template_probability_match

    # taking only 10% of histogram diff, since it's less accurate than template method
    commutative_image_diff = (img_hist_diff / 10) + img_template_diff
    return commutative_image_diff


def test_image_diff():
    # slot1 = r"examples/crops/slot1.jpg"
    # slot2 = r"examples/crops/slot2.jpg"
    # slot3 = r"examples/crops/slot3.jpg"
    # print("identical", get_image_difference(cv2.imread(slot1), cv2.imread(slot1)))
    # print("diff", get_image_difference(cv2.imread(slot1), cv2.imread(slot2)))
    # print("diff", get_image_difference(cv2.imread(slot1), cv2.imread(slot3)))

    slot1 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\mudkip_1.jpg"
    slot2 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\mudkip_2.jpg"
    slot3 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\notmudkip_1.jpg"
    print("identical", get_image_difference(cv2.imread(slot1), cv2.imread(slot1)))
    print("should match", get_image_difference(cv2.imread(slot1), cv2.imread(slot2)))
    print("should fail", get_image_difference(cv2.imread(slot1), cv2.imread(slot3)))

    # just template matching fails here
    slot1 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\allegator_8f1272f16_0.jpg"
    slot2 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\allegator_8f1a43e0f_0.jpg"
    print("should match", get_image_difference(cv2.imread(slot1), cv2.imread(slot2)))

    # just template matching fails here
    slot1 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\lusa_c26f9382d_0.jpg"
    slot2 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\lusa_c2eb92a0f_0.jpg"
    print("should match", get_image_difference(cv2.imread(slot1), cv2.imread(slot2)))

# def relative_coordinates():
#     template = r"examples/crops/faces.jpg"
#     slot1 = r"examples/crops/slot1.jpg"
#     slot2 = r"examples/crops/slot2.jpg"
#     slot3 = r"examples/crops/slot3.jpg"
#     matches = find_location_cv_multi(cv2.imread(slot1), cv2.imread(template), .5, max_count=1)
#     print(matches[0])
#     matches = find_location_cv_multi(cv2.imread(slot2), cv2.imread(template), .5, max_count=1)
#     print(matches[0])
#     matches = find_location_cv_multi(cv2.imread(slot3), cv2.imread(template), .5, max_count=1)
#     print(matches[0])


if __name__ == '__main__':
    # relative_coordinates()
    # check_matches()
    # search_for_edges("examples/1701083046739585.jpg")
    # search_for_edges("examples/11.27.23.png")
    # search_for_edges("examples/1701072300899279.jpg")
    # test_image_diff()
    process_folder("examples")
