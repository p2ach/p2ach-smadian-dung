import io, base64
from build_data import *

class_map = {
    1: 2111, 2: 2111, 3: 2121, 4: 2121,#일부 붉은 색인 경우 3에 101을 더해주어야함, 나머지 일부만 검출될 경우 100더함
    5: 1111, 6: 1111, 7: 1121, 8: 1121,#일부만 검출될 경우 100더함
    9: 3111, 10: 3111, 11: 3122, 12: 3021,#일부만 검출될 경우 100더함
    13: 4111, 14: 4111, 15: 4122, 16: 4121#일부만 검출될 경우 결과제외
}

def postprocess(im_tensor,logits):
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)

    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()
    bin_count = [bc if bc > 200 else 0 for bc in bin_count]
    count_size = 0
    for _s in bin_count:
        if _s > 0:
            count_size += 1
    list_result_outputs=[]
    if count_size==1:
        list_result_outputs.append(class_map[int(found_class_num)])
        # return list_result_output

    elif count_size > 1:
        print("bin count", bin_count)
        for i, _s in enumerate(bin_count):
            if _s > 0:
                if i == 3:
                    list_result_outputs.append(class_map[i] + 101)
                elif i < 13:
                    list_result_outputs.append(class_map[i] + 100)
    return list_result_outputs


def infer(input_json,runner):
    # print("input_json",input_json)
    img=Image.open(io.BytesIO(base64.decodebytes(input_json.encode('ascii'))))
    im_tensor = one_sample_transform_nolabel(img, None, crop_size=list(img.size).reverse(),
                                                   scale_size=(1.0, 1.0), augmentation=False)
    print("im_tensor : ", im_tensor.shape)
    im_tensor = im_tensor.to('cpu')
    logits, _ = runner.run(im_tensor.unsqueeze(0))
    list_result_outputs=postprocess(im_tensor,logits)
    print("result : ", list_result_outputs)
    return list_result_outputs





