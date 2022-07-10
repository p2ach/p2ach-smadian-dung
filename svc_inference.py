import io, base64
from build_data import *

original_class_map = {
    1: 2111, 2: 2111, 3: 2121, 4: 2121,#일부 붉은 색인 경우 3에 101을 더해주어야함, 나머지 일부만 검출될 경우 100더함
    5: 1111, 6: 1111, 7: 1121, 8: 1121,#일부만 검출될 경우 100더함
    9: 3111, 10: 3111, 11: 3122, 12: 3021,#일부만 검출될 경우 100더함
    13: 4111, 14: 4111, 15: 4122, 16: 4121#일부만 검출될 경우 결과제외
}
SPECIAL_ID=3
THRESHOLD_ID=13
class_map = {
    1: 2111, 2: 2121, 3: 2121,#일부 붉은 색인 경우 3에 101을 더해주어야함, 나머지 일부만 검출될 경우 100더함
    4: 1111, 5: 1121, 6: 1121,#일부만 검출될 경우 100더함
    7: 3111, 8: 3111, 9: 3122, 10: 3021,#일부만 검출될 경우 100더함
    11: 4111, 12: 4122, 13: 4121#일부만 검출될 경우 결과제외
}
SPECIAL_ID=2
THRESHOLD_ID=11

def filter_case(bin_count):
    if len(bin_count)<17:
        while len(bin_count)<17:
            bin_count.append(0)


    if bin_count[1] > 120 or bin_count[2] > 120:
        bin_count[3]=bin_count[4]=0

    if bin_count[5] > 120 or bin_count[6] > 120:
        bin_count[7]=bin_count[8]=0

    if bin_count[9] > 120 or bin_count[10] > 120:
        bin_count[11]=bin_count[12]=0

    if bin_count[13] > 120 or bin_count[14] > 120:
        bin_count[15]=bin_count[16]=0

    return bin_count

def postprocess(im_tensor,logits):
    logits = F.interpolate(logits, size=im_tensor.shape[1:], mode='bilinear', align_corners=True)
    max_logits, label_reco = torch.max(torch.softmax(logits, dim=1), dim=1)
    np_label_reco = label_reco[0].cpu().detach().numpy().astype(np.uint8)

    bin_count = np.bincount(np_label_reco.reshape(-1))
    bin_count[0] = 0
    found_class_num = bin_count.argmax()
    bin_count = [bc if bc > 120 else 0 for bc in bin_count]
    count_size = 0
    for _s in bin_count:
        if _s > 0:
            count_size += 1

    bin_count=filter_case(bin_count)

    list_result_outputs=[]
    if count_size==1:
        list_result_outputs.append(class_map[int(found_class_num)])
        # return list_result_output

    elif count_size > 1:
        for i, _s in enumerate(bin_count):
            if _s > 0:
                if i == SPECIAL_ID:
                    list_result_outputs.append(class_map[i] + 101)
                elif i < 11:
                    list_result_outputs.append(class_map[i] + 100)
    return list_result_outputs


def infer(input_json,runner):
    # print("input_json",input_json)
    img=Image.open(io.BytesIO(base64.decodebytes(input_json.encode('ascii'))))
    im_tensor = one_sample_transform_nolabel(img, None, crop_size=list(img.size).reverse(),
                                                   scale_size=(1.0, 1.0), augmentation=False)
    print("im_tensor : ", im_tensor.shape)
    im_tensor = im_tensor.to('cpu')
    try:
        logits, _ = runner.run(im_tensor.unsqueeze(0))
    except Exception as e:
        logits, _ = runner(im_tensor.unsqueeze(0))

    list_result_outputs=postprocess(im_tensor,logits)
    print("result : ", list_result_outputs)
    return list_result_outputs





