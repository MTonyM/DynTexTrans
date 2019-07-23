import argparse


class Options:
    def __init__(self):
        super(Options, self).__init__()
        self.parser = argparse.ArgumentParser()

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


class TrainOptions(Options):
    def __init__(self):
        super(TrainOptions, self).__init__()

        # data loader related
        self.parser.add_argument('--train_path', type=str, default='../data/dataset/',
                                 help='path of the training images')
        self.parser.add_argument('--input_dim', type=int, default=3,
                                 help='input dimensions.')

        # train related
        self.parser.add_argument('--num_keypoints', type=int, default=10,
                                 help='number of keypoints extracted from images/video frame')

        self.parser.add_argument('--outer_iter', type=int, default=50,
                                 help='number of iteration for fading in progressive training')
        self.parser.add_argument('--epoch', type=int, default=5, help='number of epoch for each outer iteration')
        self.parser.add_argument('--progressive', type=int, default=1,
                                 help='1 for using progressive training, 0 for using normal training')
        self.parser.add_argument('--batchsize', type=int, default=8,
                                 help='batchsize for level3. level3 use batchsize, level2 use 2*batchsize, level1 use 4*batchsize')
        self.parser.add_argument('--datasize', type=int, default=12800, help='number of sampled data for each epoch')
        self.parser.add_argument('--datarange', type=int, default=708,
                                 help='data sampling range for each style (data is sampled from 1.png ~ datarange.png)')
        self.parser.add_argument('--augementratio', type=float, default=0.25,
                                 help='ratio of augmented style during training')
        self.parser.add_argument('--centercropratio', type=float, default=0.5, help='ratio of center cropping')

        # model related
        self.parser.add_argument('--save_model_name', type=str, default='../save/tet.ckpt',
                                 help='specify the model name to save')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu, 0 for cpu, 1 for gpu')
        self.parser.add_argument('--inter_channels', type=int, default=256,
                                 help='number of channels in hourglass res block')
        self.parser.add_argument('--num_stack', type=int, default=3, help='number of stacks of stacked hourglasses')
        self.parser.add_argument('--ratio_drop', type=float, default=0.2,
                                 help='dropout rate of resblock in one hg reslist1.')
        self.parser.add_argument('--temperature', type=float, default=0.2,
                                 help='temperature.')
