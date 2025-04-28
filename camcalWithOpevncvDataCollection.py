import cv2
import os

def capture_checkerboard_images(num_images, save_dir):
    cap_left = cv2.VideoCapture(0)
    cap_right = cv2.VideoCapture(1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    count = 0
    while count < num_images:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Cannot capture frames.")
            break

        cv2.imshow('Left Camera', frame_left)
        cv2.imshow('Right Camera', frame_right)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            left_image_path = os.path.join(save_dir, f'left_{count}.png')
            right_image_path = os.path.join(save_dir, f'right_{count}.png')
            cv2.imwrite(left_image_path, frame_left)
            cv2.imwrite(right_image_path, frame_right)
            count += 1
            print(f"Saved pair {count}")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

# Capture 20 pairs of images
capture_checkerboard_images(20, 'checkerboard_images')
