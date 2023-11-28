import os.path
import cv2
import numpy as np
import imagehash
import tqdm
from PIL import Image

from detect_image import find_location_cv_multi, write_debug_image

IMAGE_ICON_I = r"examples/crops/icon_i.jpg"
IMAGE_LOGO = r"examples/crops/logo.jpg"
# IMAGE_BADGE = r"examples/crops/badge.jpg"
IMAGE_BOTTOM_EDGE = r"examples/crops/bottom_edge.jpg"
IMAGE_TOP_CORNER = r"examples/crops/top_corner.jpg"
NORMALIZED_FACE_SIZE = (53, 33)

SCALING_CACHE = {('examples\\11.27.23.png', 'examples/crops/logo.jpg'): (0.24, 0.9545839428901672), ('examples\\1701072300899279.jpg', 'examples/crops/logo.jpg'): (0.27, 0.9557238221168518), ('examples\\1701072740404357.png', 'examples/crops/logo.jpg'): (0.41, 0.9752677083015442), ('examples\\1701073342970426.jpg', 'examples/crops/logo.jpg'): (0.4, 0.9708191752433777), ('examples\\1701077950418456.jpg', 'examples/crops/logo.jpg'): (0.5, 0.9892277121543884), ('examples\\1701082802511688.jpg', 'examples/crops/logo.jpg'): (0.45, 0.9766697287559509), ('examples\\1701083046739585.jpg', 'examples/crops/logo.jpg'): (0.48, 0.9917872548103333), ('examples\\1701084097242461.jpg', 'examples/crops/logo.jpg'): (0.47, 0.9795607328414917), ('examples\\1701086451130592.jpg', 'examples/crops/logo.jpg'): (0.47, 0.9853900074958801), ('examples\\1701088828336338.jpg', 'examples/crops/logo.jpg'): (0.47, 0.9800499677658081), ('examples\\1701095601008634.jpg', 'examples/crops/logo.jpg'): (0.11, 0.6911693811416626), ('examples\\1701095601008634.jpg', 'examples/crops/icon_i.jpg'): (0.6458333333333334, 0.9866015315055847), ('examples\\1701096497026983.jpg', 'examples/crops/logo.jpg'): (0.5, 0.9901890754699707)}


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

    print("SCALING_CACHE", SCALING_CACHE)


def search_for_scaling(chart_image_path: str, chart_img: cv2.typing.MatLike, reference_image: str) -> (float, float):
    global SCALING_CACHE
    if (chart_image_path, reference_image) in SCALING_CACHE:
        return SCALING_CACHE[(chart_image_path, reference_image)]

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

    SCALING_CACHE[(chart_image_path, reference_image)] = best_scale, best_match
    print("SCALING_CACHE", SCALING_CACHE)
    return best_scale, best_match


def search_for_faces(chart_image_path: str) -> list[cv2.typing.MatLike]:
    chart_img = cv2.imread(chart_image_path)
    if chart_img is None:
        print("error reading chart:", chart_image_path)
        return []
    original_height, original_width, _ = chart_img.shape
    if max(original_height, original_width) > 2048:
        scale = min(2048/original_height, 2048/original_width)
        width = int(original_width * scale)
        height = int(original_height * scale)
        chart_img = cv2.resize(chart_img, (width, height), interpolation=cv2.INTER_AREA)

    best_scale, best_match = 0, 0
    for scaling_ref in [IMAGE_LOGO, IMAGE_ICON_I]:
        test_scale, test_match = search_for_scaling(chart_image_path, chart_img, scaling_ref)
        if test_match > .95:
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
    _directory, filename = os.path.split(chart_image_path)
    # out_path = os.path.join("out", "corners_" + filename)
    # write_debug_image(chart_img, matches, out_path)

    face_matches = []
    for strength, bounding_box in matches:
        face_locations = face_locations_near_match(chart_img, scale, bounding_box)
        [face_matches.append((strength, face_location)) for face_location in face_locations]
    out_path = os.path.join("out", "faces_" + filename)
    write_debug_image(chart_img, face_matches, out_path)

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
            dist1 = compute_color_dist_column(chart_img, margin_color, int(x + scale * sx) + i, int(y + scale * sy), int(y + scale * sy) + int(scale * sh))
            dist2 = compute_color_dist_column(chart_img, margin_color, int(x + scale * sx) + i + int(scale * sw), int(y + scale * sy), int(y + scale * sy) + int(scale * sh))
            # print(f"dist1 {dist1} dist2 {dist2}")
            if dist1 >= dist2:
                break
            if 45 <= dist1:
                break
        # print(f"padding by {i}")

        j = 0
        for j in range(22):
            dist2 = compute_color_dist_row(chart_img, bottom_shadow, int(x + scale * sx), int(x + scale * sx) + int(scale * sw/2), int(y + scale * sy + scale * sh) + j)
            # print(f"dist1 {dist1} dist2 {dist2}")
            # 30 causes low res images to go all the way to the end of the range; 50 causes some high res images to not budge at all
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
    distances = []
    for y in range(y_start, y_end):
        left_color = image[y, x]
        distances.append(np.linalg.norm(color - left_color))
    return sorted(distances)[int(len(distances)/2)]


def compute_color_dist_row(image: cv2.typing.MatLike, color, x_start: int, x_end, y) -> float:
    distances = []
    for x in range(x_start, x_end):
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
        best_match = 0
        for group_key in tqdm.tqdm(face_groups, f"grouping faces", disable=True):
            possible_group = face_groups[group_key]
            for group_member in possible_group:
                for x_crop, y_crop in [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)]:
                    match = cropped_image_match(face_image, group_member, x_crop, y_crop)
                    if match > best_match:
                        best_group = possible_group
                        best_match = match
                    match = cropped_image_match(group_member, face_image, x_crop, y_crop)
                    if match > best_match:
                        best_group = possible_group
                        best_match = match
        if best_group:
            best_group.append(face_image)
        else:
            phash = imagehash.phash(Image.fromarray(face_image), hash_size=6)
            face_groups[str(phash)] = [face_image]


def cropped_image_match(image1, image2, x_crop_size, y_crop_size):
    image_1_w, image_1_height, _ = image1.shape
    cropped_image_1 = image1[y_crop_size:image_1_height - 2 * y_crop_size,
                      x_crop_size:image_1_w - 2 * x_crop_size]
    matches = find_location_cv_multi(image2, cropped_image_1, .85, max_count=1)
    for strength, _box in matches:
        return strength
    return 0


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
    print("identical", cropped_image_match(cv2.imread(slot1), cv2.imread(slot1)))
    print("should match", cropped_image_match(cv2.imread(slot1), cv2.imread(slot2)))
    print("should fail", cropped_image_match(cv2.imread(slot1), cv2.imread(slot3)))

    # just template matching fails here
    slot1 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\allegator_8f1272f16_0.jpg"
    slot2 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\allegator_8f1a43e0f_0.jpg"
    print("should match", cropped_image_match(cv2.imread(slot1), cv2.imread(slot2)))

    # just template matching fails here
    slot1 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\lusa_c26f9382d_0.jpg"
    slot2 = r"Z:\ReposZ\pomas_chart\examples\similarity_tests\lusa_c2eb92a0f_0.jpg"
    print("should match", cropped_image_match(cv2.imread(slot1), cv2.imread(slot2)))

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
