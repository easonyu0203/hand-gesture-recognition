from utils.CV_Draw import Draw


def draw_result(result, holistic_result, image):
    draw = Draw()
    image = draw.draw_hand_landmark(holistic_result, image)
    if result["left"]:
        name, conf = result["left"]["name"], result["left"]["confidence"]
        is_outlier, outlier_conf = result["left"]["is_outlier"], result["left"]["outlier_confidence"]
        out_str = ("outlier" if is_outlier == -1 else "inlier") + f" {outlier_conf:2<f}"
        image = draw.draw_box(holistic_result.left_np, image, title=f"{name} {conf:<2f}, {out_str}")
    if result["right"]:
        name, conf = result["right"]["name"], result["right"]["confidence"]
        is_outlier, outlier_conf = result["right"]["is_outlier"], result["right"]["outlier_confidence"]
        out_str = ("outlier" if is_outlier == -1 else "inlier") + f" {outlier_conf:2<f}"
        image = draw.draw_box(result.holistic_result.right_np, image, title=f"{name} {conf:<2f}, {out_str}")

    return image