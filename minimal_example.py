import mccv

def trainer(args):
    print(args.pretty_print())
    print(args.pretty_print(separator="-", equal=":"))
    print(args.pretty_print(name_mapping={'learning_rate':'lr','weight_decay':'wd'}, equal="",blacklist=['dataset']))

parser = mccv.CVArgumentParser(strategy="grid_search", add_help=False)
parser.add_argument("--launch_CV", action="store_true")
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_CV_choice(
    "--learning_rate",
    default=0.1,
    type=float,
    choices=[0.2, 0.1, 0.05],
    tunable=True,
)
parser.add_CV_choice(
    "--weight_decay",
    default=1e-4,
    type=float,
    choices=[1e-5, 1e-4, 1e-3],
    tunable=True,
)
args = parser.parse_args()

if args.launch_CV:
    # loop through the first 6 configurations
    for trial_args in args.generate_trials(6):
        print(trial_args.dataset, trial_args.learning_rate, trial_args.weight_decay)
else:
    # without the launch_CV flag you can run your usual debug
    # with the user-provided/default arguments as defined above
    print(args.dataset, args.learning_rate, args.weight_decay)

trainer(args)