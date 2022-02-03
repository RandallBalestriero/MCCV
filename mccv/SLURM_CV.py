import datetime
import os


class AbstractSLURMConfig(object):

    RUN_CMD = "sbatch"
    FORBIDDEN = [
        "add_slurm_cmd",
        "add_command",
        "load_modules",
        "notify_job_status",
        "build_slurm_command",
    ]

    def __init__(self):
        now = str(datetime.datetime.now()).replace(" ", "_")
        self.log_err = now + ".err"
        self.log_out = now + ".out"
        self.modules = []
        self.time = "15:00"
        self.minutes_to_checkpoint_before_walltime = 5
        self.gpu = 1
        self.cpus_per_task = 1
        self.nodes = 1
        self.mem = 2000
        self.email = None
        self.notify_on_end = False
        self.notify_on_fail = False
        self.display_name = None
        self.srun_cmd = "python3"
        self.gpu_type = None
        self.call_load_checkpoint = False
        self.partition=None
        self.commands = []
        self.slurm_commands = []

    def add_slurm_cmd(self, cmd, value, comment):
        self.slurm_commands.append((cmd, value, comment))

    def add_command(self, cmd):
        self.commands.append(cmd)

    def load_modules(self, modules):
        self.modules = modules

    def notify_job_status(self, email, on_done, on_fail):
        self.email = email
        self.notify_on_end = on_done
        self.notify_on_fail = on_fail


class SLURMConfig(AbstractSLURMConfig):
    def __init__(self, **kwargs):
        super(SLURMConfig, self).__init__()
        for key, value in kwargs.items():
            assert key not in self.FORBIDDEN
            setattr(self, key, value)

    def build_slurm_command(self):
        sub_commands = []

        command = [
            "#!/bin/bash",
            "#",
            "# Auto-generated by SLURMCV",
            "#################\n",
        ]
        sub_commands.extend(command)

        # add job name
        command = [
            "# set a job name",
            "#SBATCH --job-name={}".format(self.display_name),
            "#################\n",
        ]
        sub_commands.extend(command)

        # add out output
        command = [
            "# a file for job output, you can check job progress",
            "#SBATCH --output={}".format(self.log_out),
            "#################\n",
        ]
        sub_commands.extend(command)

        # add err output
        command = [
            "# a file for errors",
            "#SBATCH --error={}".format(self.log_err),
            "#################\n",
        ]
        sub_commands.extend(command)

        # add job time
        command = [
            "# time needed for job",
            "#SBATCH --time={}".format(self.time),
            "#################\n",
        ]
        sub_commands.extend(command)

        if self.partition is not None:
            # add partition
            command = [
                "# partition for the job",
                "#SBATCH --partition={}".format(self.partition),
                "#################\n",
            ]
            sub_commands.extend(command)

        if self.gpu > 0:
            command = [
                "# gpus per node",
                "#SBATCH --gres=gpu:{}".format(self.gpu),
                "#################\n",
            ]
            if self.gpu_type is not None:
                command[1] = "#SBATCH --gres=gpu:{}:{}".format(self.gpu_type, self.gpu)
            sub_commands.extend(command)

        command = [
            "# cpus per job",
            "#SBATCH --cpus-per-task={}".format(self.cpus_per_task),
            "#################\n",
        ]
        sub_commands.extend(command)

        command = [
            "# number of requested nodes",
            "#SBATCH --nodes={}".format(self.nodes),
            "#################\n",
        ]
        sub_commands.extend(command)

        command = [
            "# memory per node",
            "#SBATCH --mem={}".format(self.mem),
            "#################\n",
        ]
        sub_commands.extend(command)

        command = [
            "# slurm will send a signal this far out before it kills the job",
            f"#SBATCH --signal=USR1@{self.minutes_to_checkpoint_before_walltime * 60}",
            "#################\n",
        ]

        sub_commands.extend(command)

        # Subscribe to email if requested
        mail_type = []
        if self.notify_on_end:
            mail_type.append("END")
        if self.notify_on_fail:
            mail_type.append("FAIL")
        if len(mail_type) > 0:
            mail_type_query = [
                "# Have SLURM send you an email when the job ends or fails",
                "#SBATCH --mail-type={}".format(",".join(mail_type)),
            ]
            sub_commands.extend(mail_type_query)

            email_query = [
                "#SBATCH --mail-user={}".format(self.email),
            ]
            sub_commands.extend(email_query)

        # add custom sbatch commands
        sub_commands.append("\n")
        for (cmd, value, comment) in self.slurm_commands:
            comment = "# {}".format(comment)
            cmd = "#SBATCH --{}={}".format(cmd, value)
            spaces = "#################\n"
            sub_commands.extend([comment, cmd, spaces])

        # load modules
        sub_commands.append("\n")
        for module in self.modules:
            cmd = "module load {}".format(module)
            sub_commands.append(cmd)

        # remove spaces before the hash
        sub_commands = [x.lstrip() for x in sub_commands]

        # add additional commands
        for cmd in self.commands:
            sub_commands.append(cmd)
            sub_commands.append("\n")

        # cmd = "srun {}".format(self.srun_cmd)
        sub_commands.append(self.srun_cmd)

        # build full command with empty lines in between
        full_command = "\n".join(sub_commands)
        return full_command

    def write(self, script_name=None, autolaunch=False):

        if script_name is None:
            script_name = "SLURM_CV_default.sh"
        with open(script_name, "w") as f:
            f.write(self.build_slurm_command())
        if autolaunch:
            os.system(f"sbatch {script_name}")
