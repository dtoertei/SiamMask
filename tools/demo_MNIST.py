# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob

from tools.test import *
from tools.util_mnist import * # processing the segmented image
from models.CNNmodel import * # added for loading of a CNN model


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json', help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--mnist_frames', default='../../data/fashion_MNIST/test_frames', help='path to fashion MNIST frames')
parser.add_argument('--mnist_cnn', default='../fashion_mnist_cnn/weights_InceptionCNN0.hdf5', help='path to fashion MNIST CNN weights.hdf5 file')
parser.add_argument('--save_Siam_folder', default='../../data/fashion_MNIST/test_frames_siam', help='debug: path to SiamMask segmented frames')
parser.add_argument('--save_Siam_fashion_folder', default='../../data/fashion_MNIST/test_frames_siam_fashion', help='debug: path to MNIST CNN preprocessed frames')
parser.add_argument('--font_bbox_style', default='../../data/fashion_MNIST/FiraMono-Medium.otf', help='truetype font style')
args = parser.parse_args()



if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup SiamMask Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    input_shape = (28,28,1) 
    num_classes = 10
    offset_px = 5

    # Load MNIST CNN model
    model = InceptionCNN(input_shape,num_classes)
    model.load_weights(args.mnist_cnn) # no need to compile, we'll run only the predict methods

    # Parse Image file
    img_files = sorted(glob.glob(join(args.mnist_frames, '*.pn*')))
    ims = [cv2.imread(imf) for imf in img_files]
    print(img_files)
    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()

            mask = state['mask'] > state['p'].seg_thr

            pairs = np.reshape(location, (4,2))
            pairs_max = np.max(pairs, axis=0)
            pairs_min = np.min(pairs, axis=0)

            tmp_im = im.copy()

            # for DEBUG
            #rect_pairs_display = [pairs_min[0], pairs_min[1], pairs_max[0], pairs_min[1], pairs_max[0], pairs_max[1], pairs_min[0], pairs_max[1]]
            #im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            #cv2.polylines(im, [np.int0(rect_pairs_display).reshape((-1, 1, 2))], True, (0, 0, 255), 3)

            # apply segmentation mask
            tmp_im[:, :, 0] = (mask == 1) * tmp_im[:, :, 0]
            tmp_im[:, :, 1] = (mask == 1) * tmp_im[:, :, 1]
            tmp_im[:, :, 2] = (mask == 1) * tmp_im[:, :, 2]

            skinMask = extractSkinMask(tmp_im)

            # supress skin-HSV pixels
            tmp_im[:, :, 0] = (skinMask == 1) * tmp_im[:, :, 0]
            tmp_im[:, :, 1] = (skinMask == 1) * tmp_im[:, :, 1]
            tmp_im[:, :, 2] = (skinMask == 1) * tmp_im[:, :, 2]
            
            # preprocess the segmented image for MNIST CNN
            tmp = tmp_im[int(pairs_min[1])-offset_px:int(pairs_max[1])+offset_px, int(pairs_min[0])-offset_px:int(pairs_max[0])+offset_px, :]
            input_gray = preprocess_mnist(tmp, input_shape)

            # make prediction with MNIST CNN
            predicted_class = []
            if isinstance(model, Sequential):
                predicted_class = model.predict_class(input_gray)
            else:
                scores = model.predict(input_gray)
                predicted_class = np.argmax(scores)

            # postprocess results from MNIST CNN
            colors = generate_colors(labelNames)
            # box = [ymin, xmin, ymax, xmax]
            yolo_draw = draw_boxes(Image.fromarray(im.astype('uint8'), 'RGB'), np.max(scores), [int(pairs_min[1]), int(pairs_min[0]), int(pairs_max[1]), int(pairs_max[0])], predicted_class, labelNames, colors, args.font_bbox_style)
            yolo_draw.save(img_files[f], quality=90)

            # for DEBUG
            #cv2.imwrite(os.path.join(args.save_Siam_folder,'SiamMask{}.png'.format(f)), im)
            #cv2.imwrite(os.path.join(args.save_Siam_fashion_folder,'SiamMask{}.png'.format(f)), gray)
            

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))


