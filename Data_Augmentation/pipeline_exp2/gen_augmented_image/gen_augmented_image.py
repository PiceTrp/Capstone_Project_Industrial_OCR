import traceback


def get_augment_image(text, pic_no, config): # ex: text="8A017"
    # >>> mask creation Implementation <<<
    get_character_masks(text, config)

    # >>> Insertion Implementation <<<
    fake_image_paths = sorted(glob(os.path.join(config["generated_chars_dir"], "*fake*")))
    mask_paths = sorted(glob(os.path.join(config["generated_chars_image_dir"], "*real*")))
    bg_image = np.array(Image.open(config["background_image_path"]).convert("L"))

    try:
        # get info needed of each single character
        augmented_characters = []
        for i, (fake_image_path, mask_path) in enumerate(zip(fake_image_paths, mask_paths)):
            augmented_character = get_augmented_character(fake_image_path, mask_path)
            augmented_characters.append(augmented_character)

        # get text box masked & bw_mask
        text_box_masked, text_box_bw_mask = create_text_box(augmented_characters, 
                                                            char_padding=config["char_padding"])
        # get enable-to-place area on the non-character background image
        largest_contour, placable_topleft, placable_bottomright = get_insert_area(bg_image)
        
        # randomly place text box on the non-character background image & get result
        result_image, (random_x, random_y) = randomly_place_text_box(bg_image, text_box_masked, text_box_bw_mask)

        # get bounding box of each character and get bbox of text box
        start_point = (random_x, random_y)
        text_box_h, text_box_w = text_box_masked.shape[:-1]
        text_box_xywh = [random_x, random_y, text_box_w, text_box_h]
        char_boxes_xywh = get_character_boxes_xywh(augmented_characters, start_point, config["char_padding"])

        # save result image & save annotated bbox of text box & each character
        all_output = {"result_image":result_image, "text_box_xywh":text_box_xywh, "char_boxes_xywh":char_boxes_xywh}
        save_result(config, all_output, pic_no, text)
        # print("generate and save successfully ...")
        
    except Exception as e:
        print(f"Error at {pic_no}_{text}: {e}")
        print(traceback.print_exc())
        return ""

    
    return result_image