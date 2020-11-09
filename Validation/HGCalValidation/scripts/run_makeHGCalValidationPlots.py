import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Produce root files with tracking analysis plots')
parser.add_argument('--folderin', type=str, help='Name of input folder')
parser.add_argument('--folderout', type=str, help='Name of output folder', default="~/www_eos/HGCal/HGCDoublet_validation/20201012_EMTrackSeeded/png/newFolder")
parser.add_argument('--sample', help="Data sample", type=str, choices=["pions", "kaons", "electrons", "photons", "all"])
parser.add_argument('--label',type=str, help='Label used in output log file and sub-folder')
parser.add_argument('-v', '--verbose', help="increase output verbosity", action="store_true")
args = parser.parse_args()

LABEL = args.label

pions = False
electrons = False
photons = False
print(args.sample)

PATHIN = args.folderin+'/'

def create_command(filesin, folderout, sample, label):
  final_folder = folderout + "/" + sample
  log = label+"_"+sample+'.log'
  command = 'makeHGCalValidationPlots.py '+filesin+' -o '+final_folder+' --png --collection allTiclMultiClusters >& '+log+' &'
  return command

#kaons
if (args.sample == "kaons") or (args.sample == "all") :
  SAMPLES = []
  SAMPLES.append('singleKaonL__e10GeV__nopu')
  SAMPLES.append('singleKaonL__e50GeV__nopu')
  SAMPLES.append('singleKaonL__e100GeV__nopu')
  SAMPLES.append('singleKaonL__e200GeV__nopu')
  SAMPLES.append('singleKaonL__e300GeV__nopu')
  FILESIN = ""
  for SAMPLE in SAMPLES : 
    FILESIN += PATHIN+'DQM_V0001_R000000001__step4_'+SAMPLE+'.root '
  command = create_command(FILESIN, args.folderout, "singleKaonL", args.label)
  print(command)
  os.system(command)

#photons
if (args.sample == "photons") or (args.sample == "all") :
  SAMPLES = []
  SAMPLES.append('singlephoton__e10GeV__nopu')
  SAMPLES.append('singlephoton__e50GeV__nopu')
  SAMPLES.append('singlephoton__e100GeV__nopu')
  SAMPLES.append('singlephoton__e200GeV__nopu')
  SAMPLES.append('singlephoton__e300GeV__nopu')
  FILESIN = ""
  for SAMPLE in SAMPLES : 
    FILESIN += PATHIN+'DQM_V0001_R000000001__step4_'+SAMPLE+'.root '
  command = create_command(FILESIN, args.folderout, "singlephoton", args.label)
  print(command)
  os.system(command)

#pions
if (args.sample == "pions") or (args.sample == "all") :
  SAMPLES = []
  SAMPLES.append('singlepi__e10GeV__nopu')
  SAMPLES.append('singlepi__e50GeV__nopu')
  SAMPLES.append('singlepi__e100GeV__nopu')
  SAMPLES.append('singlepi__e200GeV__nopu')
  SAMPLES.append('singlepi__e300GeV__nopu')
  FILESIN = ""
  for SAMPLE in SAMPLES : 
    FILESIN += PATHIN+'DQM_V0001_R000000001__step4_'+SAMPLE+'.root '
  command = create_command(FILESIN, args.folderout, "singlepi", args.label)
  print(command)
  os.system(command)

#electrons
if (args.sample == "electrons") or (args.sample == "all") :
  SAMPLES = []
  SAMPLES.append('singleel__e10GeV__nopu')
  SAMPLES.append('singleel__e50GeV__nopu')
  SAMPLES.append('singleel__e100GeV__nopu')
  SAMPLES.append('singleel__e200GeV__nopu')
  SAMPLES.append('singleel__e300GeV__nopu')
  FILESIN = ""
  for SAMPLE in SAMPLES : 
    FILESIN += PATHIN+'DQM_V0001_R000000001__step4_'+SAMPLE+'.root '
  command = create_command(FILESIN, args.folderout, "singleel", args.label)
  print(command)
  os.system(command)
