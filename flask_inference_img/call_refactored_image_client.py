import refactoring_image_client

test_img_path = "C:/Users/2019A00298/test_img_before_make_model/dollor_1.jpg"
MODEL_NAME = "mobilenet_money_detection"

t = refactoring_image_client.Triton(MODEL_NAME, test_img_path)
t()