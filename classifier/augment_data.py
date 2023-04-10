import os, cv2

# command to remove augmented data
# rm <path>/*_mirror* && rm <path>/*_90* && rm <path>/*_180* && rm <path>/*_270*

main_path = "dataset"
nested_paths = [
    # "whole_body/train/rusty",
    # "whole_body/train/non_rusty",

    # "whole_body/val/rusty",
    # "whole_body/val/non_rusty",

    # "whole_body/test/rusty",
    # "whole_body/test/non_rusty",

    # "thorax_with_background/train/rusty",
    # "thorax_with_background/train/non_rusty",

    # "thorax_with_background/val/rusty",
    # "thorax_with_background/val/non_rusty",

    # "thorax_with_background/test/rusty",
    # "thorax_with_background/test/non_rusty",

    # "thorax_with_background/mask_predicted_val/rusty",
    # "thorax_with_background/mask_predicted_val/non_rusty",

    # "thorax_with_background/mask_predicted_test/rusty",
    # "thorax_with_background/mask_predicted_test/non_rusty",


    # "thorax_without_background/train/rusty",
    # "thorax_without_background/train/non_rusty",

    # "thorax_without_background/val/rusty",
    # "thorax_without_background/val/non_rusty",

    # "thorax_without_background/test/rusty",
    # "thorax_without_background/test/non_rusty",

    # "thorax_without_background/mask_predicted_val/rusty",
    # "thorax_without_background/mask_predicted_val/non_rusty",
    
    # "thorax_without_background/mask_predicted_test/rusty",
    # "thorax_without_background/mask_predicted_test/non_rusty",

    "thorax_with_background/od_predicted_all_eval/rusty",
    "thorax_with_background/od_predicted_all_eval/non_rusty",
]

for nested_path in nested_paths:
    input_folder = output_folder = os.path.join(main_path, nested_path)

    # Mirror image
    for filename in os.listdir(input_folder):
        filename_parts = filename.rsplit(".", 1)
        img = cv2.imread(os.path.join(input_folder, filename))
        new_filename = filename_parts[0] + "_mirror" + "." + filename_parts[1]
        mirror = cv2.flip(img, 1)
        print(os.path.join(output_folder, new_filename))
        cv2.imwrite(os.path.join(output_folder, new_filename), mirror)


    # Rotate image
    for filename in os.listdir(input_folder):
        filename_parts = filename.rsplit(".", 1)
        img = cv2.imread(os.path.join(input_folder, filename))

        # 90 degrees rotation
        new_filename = filename_parts[0] + "_90" + "." + filename_parts[1]
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print(os.path.join(output_folder, new_filename))
        cv2.imwrite(os.path.join(output_folder, new_filename), rot)

        # 180 degrees rotation
        new_filename = filename_parts[0] + "_180" + "." + filename_parts[1]
        rot = cv2.rotate(img, cv2.ROTATE_180)
        print(os.path.join(output_folder, new_filename))
        cv2.imwrite(os.path.join(output_folder, new_filename), rot)

        # 270 degrees rotation
        new_filename = filename_parts[0] + "_270" + "." + filename_parts[1]
        rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print(os.path.join(output_folder, new_filename))
        cv2.imwrite(os.path.join(output_folder, new_filename), rot)
