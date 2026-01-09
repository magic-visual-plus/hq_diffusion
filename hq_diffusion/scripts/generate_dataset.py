
import sys
import hq_det.dataset
import numpy as np
import cv2
from tqdm import tqdm
import json


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    det_dataset = hq_det.dataset.CocoDetection(
        input_path,
        f"{input_path}/_annotations.coco.json",
        transforms=None,
    )

    output_index = 0
    output_meta = []
    for i in tqdm(range(len(det_dataset))):
        data = det_dataset[i]
        img = data['img']
        # to numpy array
        img_np = np.array(img)
        bboxes = data['bboxes']
        labels = data['cls']
        label_names = [det_dataset.id2names[l] for l in labels]
        img_id = data['image_id']

        for j in range(len(bboxes)):
            bbox = bboxes[j]
            label_name = label_names[j]

            if label_name == "裂纹":
                box_width = bbox[2] - bbox[0]
                box_height = bbox[3] - bbox[1]
                size = max(box_width, box_height)
                # make it square
                large_box_x = max(0, int(bbox[0] + box_width / 2 - size / 2))
                large_box_y = max(0, int(bbox[1] + box_height / 2 - size / 2))
                large_box_w = min(size, img_np.shape[1] - large_box_x)
                large_box_h = min(size, img_np.shape[0] - large_box_y)

                # add padding
                pad = 50
                large_box_x = max(0, large_box_x - pad)
                large_box_y = max(0, large_box_y - pad)
                large_box_w = int(min(img_np.shape[1] - large_box_x, large_box_w + 2 * pad))
                large_box_h = int(min(img_np.shape[0] - large_box_y, large_box_h + 2 * pad))

                subimg = img_np[large_box_y:large_box_y+large_box_h, large_box_x:large_box_x+large_box_w, :]
                
                image_filename = f"{output_index:05d}_img.jpg"
                output_index += 1
                cv2.imwrite(f"{output_path}/{image_filename}", cv2.cvtColor(subimg, cv2.COLOR_RGB2BGR))

                output_meta.append({
                    "image": image_filename,
                    "text": "defect of crack",
                })
                pass
            pass
        pass
    
    with open(f"{output_path}/metadata.jsonl", "w", encoding="utf-8") as f:
        for item in output_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            pass
        pass

    pass