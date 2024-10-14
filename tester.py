from lib.utils.objects import ArgClass
from model import load_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='config', default='custom_pose',
                    help='config dictionary location (default=custom_pose)')
parser.add_argument('-p', dest='phase', default='test',
                    help='network phase [train, test] (default=test)')
parser.add_argument('-l', dest='limb', default='joint',
                    help='limb [joint, bone] (default=joint)')
parser.add_argument('-s', dest='save_name', default='',
                    help='name to save the results dictionary as after training')
parsed = parser.parse_args()


print("### Libraries loaded")
# pass the argparse.Namespace object (parsed) to ArgClass to create an arg obj
arg = ArgClass(arg=parsed)

skel_model = load_model(arg)