#!/usr/bin/env python3

from __future__ import print_function
import sys,os,tempfile,shutil,subprocess,glob
import argparse

if __name__ == "__main__":

    # define options
    parser = argparse.ArgumentParser(description="Harvest track validation plots")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        help="files to be harvested (convert edm DQM format to plain ROOT format")
    parser.add_argument("-o", "--outputFile", type=str, default="harvest.root",
                        help="output file (default: 'harvest.root')")

    opts = parser.parse_args()

    # absolute path outputFile
    outputFile = os.path.abspath(opts.outputFile)

    # check the input files
    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)

    # compile a file list for cmsDriver
    filelist = ",".join(["file:{0}".format(os.path.abspath(_file)) for _file in opts.files])

    # go to a temporary directory
    _cwd = os.getcwd()
    _tempdir = tempfile.mkdtemp()
    os.chdir(_tempdir)

    # compile cmsDriver command
    cmsDriverCommand = "cmsDriver.py harvest --scenario pp --filetype DQM --conditions auto:phase2_realistic --mc -s HARVESTING:@JetMETOnlyValidation+@HGCalValidation -n -1 --filein {0}".format(filelist)
    print("# running cmsDriver" + "\n" + cmsDriverCommand)

    # run it
    subprocess.call(cmsDriverCommand.split(" "))

    # find the output and move it to the specified output file path
    ofiles = glob.glob("DQM*.root")
    if len(ofiles) != 1:
        print("ERROR: expecting exactly one output file matching DQM*.root")
        print("  ls of current directory({0}):".format(_tempdir))
        os.system("ls -lt")
        sys.exit()
    shutil.move(ofiles[0],outputFile)

    # move back to the original directory
    os.chdir(_cwd)

    # and get rid of the temporary directory
    shutil.rmtree(_tempdir)
