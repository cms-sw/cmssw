#!/usr/bin/env python

import os
import shlex
import string
import subprocess

samples = {
    'simDYtoMuMu' : {
        'datasetpath' : '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/veelken-skimGenZmumuMuPtGt15EtaLt9p9plusMuPtGt5EtaLt9p9-69987cf12dddbc8db709f491408cceae/USER', # (6864801 events)
        'dbs_url'     : 'http://cmsdbsprod.cern.ch/cms_dbs_ph_analysis_01/servlet/DBSServlet',
        'type'        : 'MC'
    }, '2012A' : {
        'datasetpath' : '/DoubleMu/aburgmei-DoubleMu_Run2012A_22Jan2013_v1_ZmumuSkim_v1-5ef1c0fd428eb740081f19333520fdc8/USER',
        'dbs_url'     : 'http://cmsdbsprod.cern.ch/cms_dbs_ph_analysis_01/servlet/DBSServlet',
        'type'        : 'Data'
    }
}

channels = {
##     'allDecayModes' : {
##         'mdtau'                        : 0,
##         'minVisibleTransverseMomentum' : "",
##      },
     'etau' : {
         'mdtau'                        : 115,
         'minVisibleTransverseMomentum' : "elec1_9had1_15"
      },
     'mutau' : {
         'mdtau'                        : 116,
         'minVisibleTransverseMomentum' : "mu1_7had1_15"
      },
##     'emu' : {
##         'mdtau'                        : 123,
##         'minVisibleTransverseMomentum' : "tau1_18tau2_8"
##      },
##     'mumu' : {
##         'mdtau'                        : 122,
##         'minVisibleTransverseMomentum' : "mu1_18mu2_8"
##      },
##     'tautau' : {
##         'mdtau'                        : 132,
##         'minVisibleTransverseMomentum' : "had1_35had2_35"
##      }
}

options = {
    'noEvtSel_embedEqRH_cleanEqDEDX_replaceGenMuons_by_%s_embedAngleEq90_noPolarization' : {
        'ZmumuCollection'              : 'genMuonsFromZs',
        'rfRotationAngle'              : 90.,        
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'gen',
        'applyMuonRadiationCorrection' : "photos",
        'cleaningMode'                 : 'DEDX',        
        'muonCaloCleaningSF'           : 1.0,
        'muonTrackCleaningMode'        : 2,
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False,
        'disableCaloNoise'             : True,
        'applyRochesterMuonCorr'       : False
    },
    'noEvtSel_embedEqRH_cleanEqDEDX_replaceRecMuons_by_%s_embedAngleEq90_noPolarization' : {
        'ZmumuCollection'              : 'goldenZmumuCandidatesGe2IsoMuons',
        'rfRotationAngle'              : 90.,        
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'rec',
        'applyMuonRadiationCorrection' : "photos",
        'cleaningMode'                 : 'DEDX',
        'muonCaloCleaningSF'           : 1.0,
        'muonTrackCleaningMode'        : 2,
        'applyZmumuSkim'               : True,
        'applyMuonRadiationFilter'     : False,
        'disableCaloNoise'             : True,
        'applyRochesterMuonCorr'       : True
    }
}

version = "v2_1_0"

crab_template = string.Template('''
[CRAB]
jobtype = cmssw
scheduler = glite
use_server = 1

[CMSSW]
datasetpath = $datasetpath
dbs_url = $dbs_url
pset = $pset
output_file = embed_AOD.root
# CV: use for MC
total_number_of_events = -1
events_per_job = 2500
# CV: use for Data
#lumi_mask = /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/DCSOnly/json_DCSONLY.txt
#total_number_of_lumis = 2
#number_of_jobs = 1
#runselection = 190450-190790

[USER]
ui_working_dir = $ui_working_dir
return_data = 0
copy_data = 1
email = christian.veelken@cern.ch 
storage_element = T2_FR_GRIF_LLR
user_remote_dir = $user_remote_dir
publish_data = 1
publish_data_name = $publish_data_name
dbs_url_for_publication = https://cmsdbsprod.cern.ch:8443/cms_dbs_ph_analysis_01_writer/servlet/DBSServlet
''')

cfg_template = "embed_cfg.py"

currentDirectory    = os.getcwd()
submissionDirectory = os.path.join(currentDirectory, "crab")

def getStringRep_bool(flag):
    retVal = None
    if flag:
        retVal = "True"
    else:
        retVal = "False"
    return retVal

def runCommand(commandLine):
    print(commandLine)
    subprocess.call(commandLine, shell = True)

crabCommands_create_and_submit = []
crabCommands_publish           = []

for sampleName, sampleOption in samples.items():
    for embeddingName, embeddingOption in options.items():
        for channelName, channelOption in channels.items():
            # CV: skip combinations that cannot be run
            if sampleOption['type'] == 'Data' and embeddingOption['replaceGenOrRecMuonMomenta'] == 'gen':
                print("Cannot run gen. Embedding on Data --> skipping sampleName = %s, embeddingName = %s" % (sampleName, embeddingName))
                continue
            
            # create config file for cmsRun
            embeddingName = embeddingName % channelName
            cfgFileName = "embed_%s_%s_cfg.py" % (sampleName, embeddingName)
            cfgFileName_full = os.path.join(submissionDirectory, cfgFileName)
            runCommand('rm -f %s' % cfgFileName_full)
            sedCommand  = "sed 's/#__//g"
            isMC = None
            if sampleOption['type'] == "Data":
                isMC = False
            elif sampleOption['type'] == "MC":
                isMC = True
            else:
                raise ValueError ("Sample = %s is of invalid type = %s !!" % (sampleName, sampleOption['type']))
            sedCommand += ";s/$isMC/%s/g" % getStringRep_bool(isMC)
            sedCommand += ";s/$ZmumuCollection/%s/g" % embeddingOption['ZmumuCollection']
            sedCommand += ";s/$mdtau/%i/g" % channelOption['mdtau']
            sedCommand += ";s/$minVisibleTransverseMomentum/%s/g" % channelOption['minVisibleTransverseMomentum'].replace(";", "\;")
            sedCommand += ";s/$rfRotationAngle/%1.0f/g" % embeddingOption['rfRotationAngle']   
            sedCommand += ";s/$embeddingMode/%s/g" % embeddingOption['embeddingMode']        
            sedCommand += ";s/$replaceGenOrRecMuonMomenta/%s/g" % embeddingOption['replaceGenOrRecMuonMomenta']
            sedCommand += ";s/$applyMuonRadiationCorrection/%s/g" % embeddingOption['applyMuonRadiationCorrection']
            sedCommand += ";s/$cleaningMode/%s/g" % embeddingOption['cleaningMode']
            sedCommand += ";s/$muonCaloCleaningSF/%f/g" % embeddingOption['muonCaloCleaningSF']
            sedCommand += ";s/$muonTrackCleaningMode/%i/g" % embeddingOption['muonTrackCleaningMode']
            sedCommand += ";s/$applyZmumuSkim/%s/g" % getStringRep_bool(embeddingOption['applyZmumuSkim'])    
            sedCommand += ";s/$applyMuonRadiationFilter/%s/g" % embeddingOption['applyMuonRadiationFilter']
            sedCommand += ";s/$disableCaloNoise/%s/g" % getStringRep_bool(embeddingOption['disableCaloNoise'])
            sedCommand += ";s/$applyRochesterMuonCorr/%s/g" % getStringRep_bool(embeddingOption['applyRochesterMuonCorr'])
            sedCommand += "'"
            sedCommand += " %s > %s" % (cfg_template, cfgFileName_full)
            runCommand(sedCommand)
            
            # create crab config file
            crabOptions = {
                'datasetpath'       : sampleOption['datasetpath'],
                'dbs_url'           : sampleOption['dbs_url'],
                'pset'              : cfgFileName_full,
                'ui_working_dir'    : os.path.join(submissionDirectory, "crabdir_%s_%s" % (sampleName, embeddingName)),
                'user_remote_dir'   : "CMSSW_5_3_x_embed_%s_%s_%s" % (sampleName, embeddingName, version),
                'publish_data_name' : "embed_%s_%s_%s" % (sampleName, embeddingName, version)
            }
            crabFileName = "crab_embed_%s_%s.cfg" % (sampleName, embeddingName)
            crabFileName_full = os.path.join(submissionDirectory, crabFileName)
            crabFile = open(crabFileName_full, 'w')
            crabConfig = crab_template.substitute(crabOptions)
            crabFile.write(crabConfig)
            crabFile.close()
    
            # keep track of commands necessary to create, submit and publish crab jobs
            crabCommands_create_and_submit.append('crab -create -cfg %s' % crabFileName_full)
            crabCommands_create_and_submit.append('crab -submit -c %s' % crabOptions['ui_working_dir'])
            
            crabCommands_publish.append('crab -publish -c %s' % crabOptions['ui_working_dir'])
    
shellFileName_create_and_submit = "embed_crab_create_and_submit.sh"
shellFile_create_and_submit = open(shellFileName_create_and_submit, "w")
for crabCommand in crabCommands_create_and_submit:
    shellFile_create_and_submit.write("%s\n" % crabCommand)
shellFile_create_and_submit.close()

shellFileName_publish = "embed_crab_publish.sh"
shellFile_publish = open(shellFileName_publish, "w")
for crabCommand in crabCommands_publish:
    shellFile_publish.write("%s\n" % crabCommand)
    shellFile_publish.write("sleep 10\n") # CV: wait for 10s to prevent overloading crab with too many commands in too short a time
shellFile_publish.close()

print("Finished building config files. Now execute 'source %s' to create & submit crab jobs." % shellFileName_create_and_submit)
print("Once all crab jobs have finished processing, execute 'source %s' to publish the samples produced." % shellFileName_publish)
