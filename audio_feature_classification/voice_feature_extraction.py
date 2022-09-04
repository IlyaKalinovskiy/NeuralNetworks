import os
import re
import subprocess

import numpy as np
import pandas as pd


class OpenSMILE:
    def __init__(self, root_dir, config_path):
        self.opensmile_dir = os.path.expanduser(root_dir)
        self.opensmile_conf = config_path
        self.verify_dependencies()

    def verify_dependencies(self):
        """Make sure we can find external dependencies, the executables ffmpeg and
        openSMILE."""
        if not os.path.exists(self.opensmile_dir):
            raise Exception(f"Can't find openSMILE home {self.opensmile_dir}")
        if not os.path.exists(self.opensmile_conf):
            raise Exception(f"Can't find openSMILE config {self.opensmile_conf}")

        try:
            command = "ffmpeg -version"
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError:
            raise Exception("Can't find ffmpeg executable")
        else:
            m = re.search(b"ffmpeg version (.*) Copyright", output, re.MULTILINE)
            if m:
                ffmpeg_version = m.group(1)
                print("ffmpeg version ", ffmpeg_version)

        try:
            command = self.opensmile_dir + "/bin/Win32/SMILExtract_Release -h"
            output = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError:
            raise Exception("Can't find SMILExtract executable")
        else:
            m = re.search(b"openSMILE version (.*)", output, re.MULTILINE)
            if m:
                opensmile_version = m.group(1)
                print("openSMILE version ", opensmile_version)

    def process(self, wav_path):
        ##----------------------------------------------------------
        ## process audio files
        ##----------------------------------------------------------
        # print("processing:", wav_path)

        ##----------------------------------------------------------
        ## extract features with openSMILE
        ## example: SMILExtract -I output.wav -C ./openSMILE-2.1.0/config/gemaps/GeMAPSv01a.conf --csvoutput features.csv
        ##----------------------------------------------------------
        features_file = os.path.dirname(wav_path) + "/temp.csv"
        command = "{opensmile_dir}/bin/Win32/SMILExtract_Release -I {input_file} -C {conf_file} --csvoutput {output_file}".format(
            opensmile_dir=self.opensmile_dir,
            input_file=wav_path,
            conf_file=self.opensmile_conf,
            output_file=features_file,
        )
        command = command.replace("\\", "/")
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

        ##----------------------------------------------------------
        ## merge metadata and features
        ##----------------------------------------------------------
        features = pd.read_csv(features_file, sep=";", index_col=None)
        ## get rid of useless column
        features.drop("name", axis=1, inplace=True)
        ## force the indexes to be equal so they will concat into 1 row. WTF, pandas?
        features = np.transpose(features.values[0])

        if os.path.exists(features_file):
            os.remove(features_file)

        # print("processing complete!")
        return features
