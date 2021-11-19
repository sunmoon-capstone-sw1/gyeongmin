import refactoring_image_client_for_yolo_2 as yolo

test_img_path = "C:/Users/2019A00298/yolo_img.png"
model_name = "yolo_money_detection"

t = yolo.Triton(model_name, test_img_path)
t()