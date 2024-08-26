from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--done', type=str, default="False",
                    help='is the exeriment finished')
args = parser.parse_args()

with open('experiment_completed_VAR.txt', 'w') as file:
    file.write(args.done)