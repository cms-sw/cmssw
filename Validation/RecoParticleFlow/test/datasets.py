#!/usr/bin/env python
from __future__ import print_function

#Hack to get ROOT to ignore command line arguments that we want
#to pass to Python

def import_ROOT():
    import sys
    tmpargv = sys.argv
    sys.argv = ['-b', '-n']
    import ROOT
    sys.argv[:] = tmpargv[:]
    return ROOT

import yaml
import subprocess
import logging
import json
import argparse
import glob
import multiprocessing
import optparse
import shlex
import os

LOG_MODULE_NAME = logging.getLogger(__name__)

class Dataset:

    """Datatype that represents a DAS dataset

    Attributes:
        global_file_prefix (string): The ROOT TFile prefix that allows to open an LFN (/store/...)
        name (string): The DAS name of the dataset
        process (string): The nickname for the physics process that this dataset belongs to
    """

    def __init__(self, name, process, global_file_prefix, cache_location, use_cache, tmpdir):
        """Summary

        Args:
            name (string): The DAS name of the dataset
            process (string): The nickname for the physics process that this dataset belongs to
            global_file_prefix (string): The ROOT TFile prefix that allows to open an LFN (/store/...)
            cache_location (string): The location of the local file cache
            use_cache (boolean): If true, access files from cache_location instead of global_file_prefix in jobs
        """
        self.name = name
        self.process = process
        self.global_file_prefix = global_file_prefix
        self.cache_location = cache_location
        self.use_cache = use_cache
        self.tmpdir = tmpdir
        self.files = None
        self.max_files = None

    def __repr__(self):
        """

        Returns:
            string: The string representation of the Dataset
        """
        s = "Dataset(name={0})".format(self.name)
        return s

    def escape_name(self):
        """Removes any slashes and other characters from the name such that it can be used as a filename

        Returns:
            string: The DAS name usable as a filename
        """
        name = self.name.replace("/", "__")
        if name.startswith("__"):
            name = name[2:]
        return name

    def get_das_cache_filename(self):
        """Summary

        Returns:
            TYPE: Description
        """

        return os.path.join(self.tmpdir, "das_cache", self.process + ".txt")
        #return os.path.join(self.tmpdir, "das_cache", self.process + ".txt", self.escape_name() + ".txt")

    def get_filenames(self):
        """Summary

        Args:
            njob (TYPE): Description

        Returns:
            TYPE: Description
        """
        ret = None
        with open(self.get_das_cache_filename(), "r") as fi:
            ret = [self.global_file_prefix + li.strip() for li in fi.readlines()]
        return ret

    def cache_das_filenames(self):
        """Summary

        Returns:
            TYPE: Description
        """
        LOG_MODULE_NAME.info("caching dataset {0}".format(self.name))
        ret = subprocess.check_output('dasgoclient --query="file dataset={0}" --limit=0'.format(self.name), shell=True)

        target_dir = os.path.dirname(self.get_das_cache_filename())
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        nfiles = 0
        with open(self.get_das_cache_filename(), "w") as fi:
            for line in ret.split("\n"):
                if line.endswith(".root"):
                    fi.write(self.global_file_prefix + line + "\n")
                    nfiles += 1

        LOG_MODULE_NAME.info("retrieved {0} files from DAS".format(nfiles))

        return

if __name__ == "__main__":

    #prefix = ""
    prefix = "root://cmsxrootd.fnal.gov//"
    #prefix = "root://xrootd-cms.infn.it//"
    tmpdir = "tmp"
    datasets = [
        Dataset("/RelValQCD_FlatPt_15_3000HS_14/CMSSW_11_2_0_pre6-112X_mcRun3_2021_realistic_v7-v1/GEN-SIM-DIGI-RAW", "QCD_noPU", prefix, None, False, tmpdir),
        Dataset("/RelValQCD_FlatPt_15_3000HS_14/CMSSW_11_2_0_pre6-PU_112X_mcRun3_2021_realistic_v7-v1/GEN-SIM-DIGI-RAW", "QCD_PU", prefix, None, False, tmpdir),
        Dataset("/RelValZEE_14/CMSSW_11_2_0_pre6-PU_112X_mcRun3_2021_realistic_v7-v1/GEN-SIM-DIGI-RAW", "ZEE_PU", prefix, None, False, tmpdir),
        Dataset("/RelValZMM_14/CMSSW_11_2_0_pre6-PU_112X_mcRun3_2021_realistic_v7-v1/GEN-SIM-DIGI-RAW", "ZMM_PU", prefix, None, False, tmpdir),
        Dataset("/RelValTenTau_15_500/CMSSW_11_2_0_pre6-PU_112X_mcRun3_2021_realistic_v7-v1/GEN-SIM-DIGI-RAW", "TenTau_PU", prefix, None, False, tmpdir),
        Dataset("/RelValNuGun/CMSSW_11_2_0_pre6-PU_112X_mcRun3_2021_realistic_v7-v1/GEN-SIM-DIGI-RAW", "NuGun_PU", prefix, None, False, tmpdir)]
    for ds in datasets:
        ds.cache_das_filenames()
