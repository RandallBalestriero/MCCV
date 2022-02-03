import argparse
from . import CV_sampler
import math
from argparse import ArgumentParser
from copy import deepcopy
from gettext import gettext as _
import re
import numpy as np
import os
import itertools

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CVArgumentParser(ArgumentParser):
    """
    Subclass of argparse ArgumentParser which adds optional calls to sample from lists or ranges
    Also enables running optimizations across parallel processes
    """

    def __init__(self, strategy="grid_search", **kwargs):
        """

        :param strategy: 'grid_search', 'random_search'
        :param enabled:
        :param experiment:
        :param kwargs:
        """
        ArgumentParser.__init__(self, **kwargs)

        self.strategy = strategy
        self.trials = []
        self.parsed_args = None
        self.opt_args = {}
        self.json_config_arg_name = None
        self.pool = None

    def add_argument(self, *args,**kwargs):
        if "action" in kwargs:
            if kwargs['action'] =="store_true":
                del kwargs['action']
                super().add_argument(*args, type=str2bool, nargs='?',
                                const=True, default=False,
                                **kwargs)
                return
        super().add_argument(*args,**kwargs)


    def add_CV_choice(self, *args, choices=None, tunable=False, **kwargs):
        self.add_argument(*args, **kwargs)
        for i in range(len(args)):
            arg_name = args[i]
            self.opt_args[arg_name] = OptArg(
                arg_name=arg_name, options=choices, tunable=tunable
            )

    def add_CV_range(
        self,
        *args,
        low,
        high,
        nb_samples=10,
        tunable=False,
        log_base=None,
        **kwargs,
    ):
        arg_type = kwargs["type"]
        self.add_argument(*args, **kwargs)
        arg_name = args[-1]
        self.opt_args[arg_name] = OptArg(
            arg_name=arg_name,
            options=[low, high],
            arg_type=arg_type,
            nb_samples=nb_samples,
            tunable=tunable,
            log_base=log_base,
        )

    def parse_args(self, args=None, namespace=None):
        # call superclass arg first
        # allow bypassing certain missing params which other parts of test tube may introduce
        results, argv = self.parse_known_args(args, namespace)

        assert "launch_CV" in results

        # extract vals
        old_args = vars(results)

        # track args
        self.parsed_args = deepcopy(old_args)
        # attach optimization fx
        old_args["trials"] = self.opt_trials
        old_args["generate_trials"] = self.generate_trials
        old_args["max_trials"] = len(
            list(itertools.product(*self.__flatten_params(self.opt_args)))
        )

        return TTNamespace(**old_args)

    def opt_trials(self, nb_trials=None):
        self.trials = CV_sampler.generate_trials(
            strategy=self.strategy,
            flat_params=self.__flatten_params(self.opt_args),
            nb_trials=nb_trials,
        )

        for trial in self.trials:
            ns = self.__namespace_from_trial(trial)
            yield ns

    def generate_trials(self, nb_trials=None):
        trials = CV_sampler.generate_trials(
            strategy=self.strategy,
            flat_params=self.__flatten_params(self.opt_args),
            nb_trials=nb_trials,
        )

        trials = [self.__namespace_from_trial(x) for x in trials]
        return trials

    def __namespace_from_trial(self, trial):
        trial_dict = {d["name"]: d["val"] for d in trial if d["name"] != "launch_CV"}
        for k, v in self.parsed_args.items():
            if k not in trial_dict:
                trial_dict[k] = v

        return TTNamespace(**trial_dict)

    def __flatten_params(self, params):
        """
        Turns a list of parameters with values into a flat tuple list of lists
        so we can permute
        :param params:
        :return:
        """
        flat_params = []
        for i, (opt_name, opt_arg) in enumerate(params.items()):
            if opt_arg.tunable:
                clean_name = opt_name.strip("-")
                clean_name = re.sub("-", "_", clean_name)
                param_groups = []
                for val in opt_arg.options:
                    param_groups.append({"idx": i, "val": val, "name": clean_name})
                flat_params.append(param_groups)
        return flat_params


class TTNamespace(argparse.Namespace):
    def __str__(self):
        result = "-" * 100 + "\nHyperparameters:\n"
        for k, v in self.__dict__.items():
            result += "{0:20}: {1}\n".format(k, v)
        return result

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()

        # remove all functions from the namespace
        clean_state = {}
        for k, v in state.items():
            if not hasattr(v, "__call__"):
                clean_state[k] = v

        # what we return here will be stored in the pickle
        return clean_state

    def __setstate__(self, newstate):
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(newstate)

    def format_args(self, blacklist=None):
        if blacklist is None:
            blacklist = ["launch_CV"]
        else:
            blacklist += ["launch_CV"]

        command = " ".join(
            [f"--{k} {v}" for (k, v) in self.__dict__.items() if k not in blacklist]
        )
        return command

    def pretty_print(self,name_mapping=None,blacklist=None,separator="_",equal="-"):
        if name_mapping is None:
            name_mapping = {}
        if blacklist is None:
            _blacklist = ["generate_trials", "max_trials", "trials", "launch_CV"]
        else:
            _blacklist = blacklist + ["generate_trials", "max_trials", "trials", "launch_CV"]
        
        assert type(separator) == str
        args = []
        for (k, v) in self.__dict__.items():
            if k not in _blacklist:
                if k in name_mapping:
                    args.append(f"{name_mapping[k]}{equal}{v}")
                else:
                    args.append(f"{k}{equal}{v}")
        return separator.join(args)



class OptArg(object):
    def __init__(
        self,
        arg_name,
        options,
        arg_type=None,
        nb_samples=None,
        tunable=False,
        log_base=None,
    ):
        self.options = options
        self.arg_name = arg_name
        self.tunable = tunable

        # convert range to list of values
        if nb_samples:
            low, high = options

            if log_base is None:
                # random search on uniform scale
                if arg_type is int:
                    self.options = np.random.choice(
                        np.arange(low, high), nb_samples, replace=False
                    )
                elif arg_type is float:
                    self.options = np.random.uniform(low, high, nb_samples)
            else:
                # random search on log scale with specified base
                assert (
                    high >= low > 0
                ), "`options` must be positive to do log-scale search."

                log_low, log_high = math.log(low, log_base), math.log(high, log_base)

                self.options = log_base ** np.random.uniform(
                    log_low, log_high, nb_samples
                )
