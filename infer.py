import numpy as np

from common.arguments import parse_args
import torch

import os

from common.camera import *
from common.model import *
from common.generators import UnchunkedGenerator


if __name__ == '__main__':
    args = parse_args()
    print(args)

    dataset = np.load(args.dataset, allow_pickle=True).item()
    metadata = dataset['metadata']

    input_keypoints = normalize_screen_coordinates(dataset['keypoints'][..., :2], w=metadata['w'], h=metadata['h'])

    filter_widths = [int(x) for x in args.architecture.split(',')]

    kps_left, kps_right = [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]
    joints_left, joints_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

    model_pos = TemporalModel(input_keypoints.shape[-2], input_keypoints.shape[-1], input_keypoints.shape[-2],
                              filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                              dense=args.dense)

    receptive_field = model_pos.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    if args.causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    if args.resume or args.evaluate:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos.load_state_dict(checkpoint['model_pos'])


    def evaluate(test_generator):
        with torch.no_grad():
            model_pos.eval()
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()

                # Positional model
                predicted_3d_pos = model_pos(inputs_2d)

                # Test-time augmentation (if enabled)
                if test_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

                return predicted_3d_pos.squeeze(0).cpu().numpy()


    print('Rendering...')

    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen)

    print('Exporting joint positions to', args.viz_export)
    # Predictions are in camera space
    np.save(args.viz_export, prediction)
