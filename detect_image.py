import cv2


def find_location_cv_multi(template_img: cv2.typing.MatLike, scene_img: cv2.typing.MatLike, threshold=.5, max_count=10):
    t_height, t_width, t_channels = template_img.shape
    s_height, s_width, s_channels = scene_img.shape
    assert t_height > 0 and t_width > 0, "template image shouldn't be empty"
    assert s_height > 0 and s_width > 0, "scene image shouldn't be empty"

    result = cv2.matchTemplate(scene_img, template_img, cv2.TM_CCOEFF_NORMED)
    h, w = template_img.shape[:2]

    strengths_and_bounding_boxes = []
    count = 0
    while count < max_count:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            break
        x, y = max_loc
        strengths_and_bounding_boxes.append((max_val, (x, y, t_width, t_height)))
        # print("match (str {}) at x,y ({}, {}) with size ({}, {})".format(max_val, x, y, w, h))
        result[max_loc[1] - h // 2:max_loc[1] + h // 2 + 1, max_loc[0] - w // 2:max_loc[0] + w // 2 + 1] = 0
        count += 1
    return strengths_and_bounding_boxes


def write_debug_image(image: cv2.typing.MatLike,
                      strengths_and_bounding_boxes: list[tuple[str, tuple[int, int, int, int]]], out_file_path: str):
    source = image
    for strength, bounding_box in strengths_and_bounding_boxes:
        x, y, width, height = bounding_box
        cv2.rectangle(source, (x, y), (x + width, y + height), (int(strength * 255), 0, 0))
    cv2.imwrite(out_file_path, source)
