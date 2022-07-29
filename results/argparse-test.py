import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train, evaluate, and store data from convolutional models')
    parser.add_argument(
        '-m',
        required=True,
        choices=['lenet', 'vgg'],
        help='Model category')
    parser.add_argument(
        '-c',
        required=True,
        choices=['i', 'n'],
        help='Category this experiment is classified under')
    parser.add_argument(
        '-i',
        required=True,
        choices=[
            'zeros', 'ones', 'normal', 'uniform', 'xavier_normal',
            'xavier_uniform', 'kaiming_uniform'],
        help='Initialization technique')
    parser.add_argument(
        '-n',
        required=True,
        choices=['nn', 'bn', 'ln', 'gn'],
        help='Normalization technique')
    parser.add_argument(
        '-f',
        choices=['in', 'out'],
        help='Fan in or fan out, only for kaiming uniform initialization')
    parser.add_argument(
        '--numeric',
        action='store_true',
        help='Store parameters, gradients, and loss to file')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Write summary and accuracy loss to file')
    parser.add_argument(
        '--print',
        action='store_true',
        help='Print model specifications to terminal')

    args = vars(parser.parse_args())
    category = 'weightinit' if args['c'] == 'i' else 'weightnorm'
    base_location = -1

    if args['i'] != 'kaiming_uniform' and args['f'] != None:
        raise Exception(
            "Fan in or fan out only allowed with Kaiming Uniform initialization")
    if args['i'] == 'kaiming_uniform' and args['f'] == None:
        raise Exception(
            "Kaiming Uniform initalization requires specification of fan in or fan out")
    if args['numeric'] or args['summary']:
        import os
        base_location = 'experiments/' + category + '/' + args['m'] + '-' + \
            args['i'] + '-' + args['n'] + '/'
        try:
            os.mkdir(base_location)
        except FileNotFoundError:
            os.makedirs(base_location)
    if args['print']:
        print("Category: " + category)
        print("Model: " + args['m'])
        print("Init: " + args['i'])
        print("Norm: " + args['n'])

    return args, base_location


def main():
    args, base_location = parse_args()
    breakpoint()


if __name__ == '__main__':
    main()
