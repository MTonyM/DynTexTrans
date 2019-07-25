import os

import cv2

if __name__ == '__main__':
    data_root = '/Users/tony/PycharmProjects/DynTexTrans/data/raw/flame'
    data_to_crop = sorted(os.listdir(data_root), key=lambda x: int(x.split('.')[0]))
    single_frame = cv2.imread(os.path.join(data_root, '184.png'), cv2.IMREAD_UNCHANGED)
    crop_cord = [20, 246, 45, 289]
    u, d, l, r = crop_cord

    for i in range(60, 211):
        single_frame = cv2.imread(os.path.join(data_root, '{}.png'.format(i)), cv2.IMREAD_UNCHANGED)
        successed = cv2.imwrite(
            os.path.join('/Users/tony/PycharmProjects/DynTexTrans/data/processed', '{}.png'.format(i - 60)),
            cv2.resize(single_frame[u:d, l:r, :], (64, 64)))
