import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train, evaluate, and store data from convolutional models')
    parser.add_argument(
        '--model',
        choices=['LeNet', 'VGG'],
        help='Model category')
    parser.add_argument(
        '--init',
        choices=[
            'zeros', 'ones', 'random', 'uniform', 'xavier_random',
            'xavier_uniform', 'kaiming_uniform'],
        help='Initialization technique')
    parser.add_argument(
        '--norm',
        choices=['nn', 'bn', 'ln', 'gn'],
        help='Normalization technique')
    parser.add_argument(
        '--fan',
        choices=['in', 'out'],
        help='Fan in or fan out, only for kaiming uniform initialization')
    parser.add_argument(
        '--numeric',
        action='store_true',
        help='Store parameters and gradients')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Write summary and accuracy loss to file')
    parser.add_argument(
        '--print',
        action='store_true',
        help='Print model specifications to terminal')

    args = vars(parser.parse_args())

    if args['model'] == None:
        raise Exception("Must specify model category")

    if args['init'] == None:
        raise Exception("Must specify initialization technique")

    if args['norm'] == None:
        raise Exception("Must specify normalization technique")

    if args['init'] != 'kaiming_uniform' and args['fan'] != None:
        raise Exception(
            "Fan in or fan out only allowed with Kaiming Uniform initialization")

    if args['init'] == 'kaiming_uniform' and args['fan'] == None:
        raise Exception(
            "Kaiming Uniform initalization requires specification of fan in or fan out")

    if args['print']:
        print("Model: " + args['model'])
        print("Init: " + args['init'])
        print("Norm: " + args['norm'])

    return args


def main():
    args = parse_args()


if __name__ == '__main__':
    main()
