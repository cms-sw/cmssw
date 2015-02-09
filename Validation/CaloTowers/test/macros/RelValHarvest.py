#!/usr/bin/python

import sys, os

from optparse import OptionParser # Command line parsing
usage = "usage: %prog summary files"
version = "%prog."
parser = OptionParser(usage=usage,version=version)
parser.add_option("-p", "--printDataSets", action="store_true", dest="printDS", default=False, help="Print datasets without attempting to download.")
(options, args) = parser.parse_args()

# This is a dictionary of flags to pull out the datasets of interest mapped to the desired name from the hcal script
dsFlags = {'RelValTTbar_':'TTbar', 'RelValQCD_Pt_80_120_':'QCD', 'RelValQCD_Pt_3000_3500_':'HighPtQCD', 'RelValMinBias_':'MinBias'}
# Dataset to select (simply so we only get one of each sample above)
sds = "GEN-SIM-RECO"
# filename prefix 
fnPrefix = "DQM_V0001_R000000001"
# blank curl command 
curl = "/usr/bin/curl -O -L --capath %(CERT_DIR)s --key %(USER_PROXY)s --cert %(USER_PROXY)s https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/%(relvalDIR)s/%(fname)s"
# output file name blank
ofnBlank = "HcalRecHitValidationRelVal_%(sample)s_%(label)s_%(info)s.root"
# default release file for MC stub
dfTextFile = "%s_%s.txt"

# ensure all required parameters are included
if len(args) < 2:
    print "Usage: ./RelValHarvest.py fullReleaseName [RelValDataSetList.txt] [Dataset]"
    print "fullReleaseName : CMSSW_5_2_0_pre3"
    print "RelValDataSetList.txt : text file from relval announcement"
    print "Dataset : usually DQMIO, but if that is not avaliable then GEN-SIM-RECO"
    exit(0)

# gather input parameter
label     = args[0]
fileinPost = "standard"
dataset = "DQMIO"
if len(args) > 1:
    fileinPost = args[1]
    if len(args) > 2:
        dataset = args[2]

filein = dfTextFile%(label, fileinPost)
print "Taking filenames form file %s"%filein

# retrieve the list of datasets
if not os.path.isfile(filein):
    os.system("wget http://cms-project-relval.web.cern.ch/cms-project-relval/relval_stats/%s"%filein)

# modify label to shortened format (remove CMSSW and '_')
slabel = label.replace('CMSSW','').replace("_","")

# get relval dir from label
clabel = label.split("_")
relvalDIR = "%s_%s_%s_x"%(clabel[0], clabel[1], clabel[2])

# initialize voms proxy for transfer
if not options.printDS:
    os.system('voms-proxy-init -voms cms')

# gather necessary proxy info for curl
X509_CERT_DIR = os.getenv('X509_CERT_DIR', "/etc/grid-security/certificates")
X509_USER_PROXY = os.getenv('X509_USER_PROXY')

# open raw inpout file 
fin = open(filein, "r")

# loop over file and pull out lines of interest
for line in fin:
    # limit to one entry per dataset
    if sds in line:
        # select datasets of interest
        for str in dsFlags.keys():
            if str in line:
                # extract dataset path
                path = line.split('|')[1].strip()
                #print "Getting DQM output from dataset: %s"%path
                print path
                if options.printDS:
                    continue
                # construct file name
                fname = fnPrefix + path.replace("/","__").replace(sds, "%s.root"%dataset)
                # copy file with curl
                curlCommand = curl%{"CERT_DIR":X509_CERT_DIR,"USER_PROXY":X509_USER_PROXY, "relvalDIR":relvalDIR,"fname":fname}
                print curlCommand
                os.system(curlCommand)
                # rename file for use with hcal scripts
                # DQM_V0001_R000000001__RelValTTbar_13__CMSSW_7_2_0_pre7-PU50ns_PRE_LS172_V12-v1__DQMIO.root
                # get dataset info from file name 
                info = fname.split("__")[2].replace(label, "").strip("-")
                ofn = ofnBlank%{"sample":dsFlags[str],"label":slabel,"info":info}
                mvCommand = "mv %(fn)s %(ofn)s"%{"fn":fname,"ofn":ofn}
                print mvCommand
                os.system(mvCommand)
                print ""

if options.printDS:
    exit()

# Copy the single pion scan part from Salavat's directory
spFileName = "pi50scan%s_ECALHCAL_CaloTowers.root"%slabel
cpCommand = "cp /afs/cern.ch/user/a/abdullin/public/pi50_scan/%s ."%spFileName
if not os.path.isfile(spFileName):
    print cpCommand
    os.system(cpCommand)
print ""
