#!/usr/bin/python

import sys, os

from optparse import OptionParser # Command line parsing
usage = "usage: %prog summary files"
version = "%prog."
parser = OptionParser(usage=usage,version=version)
parser.add_option("-p", "--printDataSets", action="store_true", dest="printDS", default=False, help="Print datasets without attempting to download.")
(options, args) = parser.parse_args()

# This is a dictionary of flags to pull out the datasets of interest mapped to the desired name from the hcal script
dsFlags = {'RelValTTbar_13__':'TTbar', 'RelValQCD_Pt_80_120_13__':'QCD', 'RelValQCD_Pt_3000_3500_13__':'HighPtQCD', 'RelValMinBias_13TeV_pythia8__':'MinBias'}
# filename prefix 
fnPrefix = "DQM_V0001_R000000001"
# blank curl command 
curl = "/usr/bin/curl -O -L --capath %(CERT_DIR)s --key %(USER_PROXY)s --cert %(USER_PROXY)s https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/%(relvalDIR)s"
# output file name blank
ofnBlank = "HcalRecHitValidationRelVal_%(sample)s_%(label)s_%(info)s.root"
# default release file for MC stub
#dfTextFile = "%s_%s.txt"
dfTextFile = "%s"

# ensure all required parameters are included
if len(args) < 1:
    print "Usage: ./RelValHarvest.py fullReleaseName"
    print "fullReleaseName : CMSSW_7_4_0_pre8"
    exit(0)

# gather input parameter
label     = args[0]

# gather necessary proxy info for curl
X509_CERT_DIR = os.getenv('X509_CERT_DIR', "/etc/grid-security/certificates")
X509_USER_PROXY = os.getenv('X509_USER_PROXY')

# modify label to shortened format (remove CMSSW and '_')
slabel = label.replace('CMSSW','').replace("_","")

# get relval dir from label
clabel = label.split("_")
relvalDIR = "%s_%s_%s_x"%(clabel[0], clabel[1], clabel[2])

print "Taking filenames form file %s"%relvalDIR

# retrieve the list of datasets
if not os.path.isfile(relvalDIR):
    #os.system("wget http://cms-project-relval.web.cern.ch/cms-project-relval/relval_stats/%s"%filein)
    curlCommand = curl%{"CERT_DIR":X509_CERT_DIR, "USER_PROXY":X509_USER_PROXY, "relvalDIR":relvalDIR}
    print curlCommand
    os.system(curlCommand)        

# initialize voms proxy for transfer
#if not options.printDS:
#    os.system('voms-proxy-init -voms cms')

# open raw inpout file 
fin = open(relvalDIR, "r")

# loop over file and pull out lines of interest
for line in fin:
    # limit to one entry per dataset
    if label in line:
        # select datasets of interest
        for str in dsFlags.keys():
            if str in line:
                # extract dataset path
                path = line.split('\'')[1].strip()

                #print "Getting DQM output from dataset: %s"%path
                print path
                if options.printDS:
                    continue

                # construct file name
                fname = path.split("/")[-1]

                # create file name for use with hcal scripts
                info = fname.split("__")[2].replace(label, "").strip("-")
                ofn = ofnBlank%{"sample":dsFlags[str],"label":slabel,"info":info}
                
                #Check if file exists already
                if not os.path.isfile(ofn):
                    # copy file with curl
                    curlCommand = curl%{"CERT_DIR":X509_CERT_DIR,"USER_PROXY":X509_USER_PROXY, "relvalDIR":relvalDIR} + "/" + fname
                    print curlCommand
                    os.system(curlCommand)

                    # Rename file for use with HCAL scripts
                    mvCommand = "mv %(fn)s %(ofn)s"%{"fn":fname,"ofn":ofn}
                    print mvCommand
                    os.system(mvCommand)
                    print ""

if options.printDS:
    exit()

# Copy the single pion scan part from Salavat's directory
spFileName = "pi50scan%s_fullGeom_ECALHCAL_CaloTowers.root"%slabel
cpCommand = "cp /afs/cern.ch/user/a/abdullin/public/pi50_scan/%s ."%spFileName
if not os.path.isfile(spFileName):
    print cpCommand
    os.system(cpCommand)
print ""
