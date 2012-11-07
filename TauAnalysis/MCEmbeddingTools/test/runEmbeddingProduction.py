#!/usr/bin/env python

import os
import shlex
import string
import subprocess

samples = {
    'simDYtoMuMu' : '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/Summer12_DR53X-PU_S10_START53_V7A-v2/GEN-SIM-RECO'
}

version = "v1_0"

options = {
    # e+tau samples
    'noEvtSel_embedEqPF_replaceGenMuons_by_etau' : {
        'mdtau'                        : 115,
        'minVisibleTransverseMomentum' : "elec1_9_had1_15",
        'embeddingMode'                : 'PF',
        'replaceGenOrRecMuonMomenta'   : 'gen',
        'cleaningMode'                 : 'PF',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    },
    'noEvtSel_embedEqRH_cleanEqPF_replaceGenMuons_by_etau' : {
        'mdtau'                        : 115,
        'minVisibleTransverseMomentum' : "elec1_9_had1_15",
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'gen',
        'cleaningMode'                 : 'PF',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    },
    'noEvtSel_embedEqRH_cleanEqDEDX_replaceGenMuons_by_etau' : {
        'mdtau'                        : 115,
        'minVisibleTransverseMomentum' : "elec1_9_had1_15",
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'gen',
        'cleaningMode'                 : 'DEDX',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    },
    'noEvtSel_embedEqRH_cleanEqDEDX_replaceRecMuons_by_etau' : {
        'mdtau'                        : 115,
        'minVisibleTransverseMomentum' : "elec1_9_had1_15",
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'rec',
        'cleaningMode'                 : 'DEDX',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    },
    # mu+tau samples
    'noEvtSel_embedEqPF_replaceGenMuons_by_mutau' : {
        'mdtau'                        : 116,
        'minVisibleTransverseMomentum' : "mu1_6_had1_15",
        'embeddingMode'                : 'PF',
        'replaceGenOrRecMuonMomenta'   : 'gen',
        'cleaningMode'                 : 'PF',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    },
    'noEvtSel_embedEqRH_cleanEqPF_replaceGenMuons_by_mutau' : {
        'mdtau'                        : 116,
        'minVisibleTransverseMomentum' : "mu1_6_had1_15",
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'gen',
        'cleaningMode'                 : 'PF',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    },
    'noEvtSel_embedEqRH_cleanEqDEDX_replaceGenMuons_by_mutau' : {
        'mdtau'                        : 116,
        'minVisibleTransverseMomentum' : "mu1_6_had1_15",
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'gen',
        'cleaningMode'                 : 'DEDX',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    },
    'noEvtSel_embedEqRH_cleanEqDEDX_replaceRecMuons_by_mutau' : {
        'mdtau'                        : 116,
        'minVisibleTransverseMomentum' : "mu1_6_had1_15",
        'embeddingMode'                : 'RH',
        'replaceGenOrRecMuonMomenta'   : 'rec',
        'cleaningMode'                 : 'DEDX',
        'applyZmumuSkim'               : False,
        'applyMuonRadiationFilter'     : False
    }
}

crab_template = string.Template('''
[CRAB]
jobtype = cmssw
scheduler = glite
use_server = 1

[CMSSW]
datasetpath = $datasetpath
pset = $pset
output_file = embed_AOD.root
# CV: use for MC
total_number_of_events = -1
events_per_job = 20000
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

for sampleName, datasetpath in samples.items():
    for label, option in options.items():
        # create config file for cmsRun
        cfgFileName = "embed_%s_%s_cfg.py" % (sampleName, label)
        runCommand('rm -f %s' % cfgFileName)
        sedCommand  = "sed 's/#__//g"
        sedCommand += ";s/$mdtau/%i/g" % option['mdtau']
        sedCommand += ";s/$minVisibleTransverseMomentum/%s/g" % option['minVisibleTransverseMomentum'].replace(";", "\;")
        sedCommand += ";s/$embeddingMode/%s/g" % option['embeddingMode']
        sedCommand += ";s/$replaceGenOrRecMuonMomenta/%s/g" % option['replaceGenOrRecMuonMomenta']
        sedCommand += ";s/$cleaningMode/%s/g" % option['cleaningMode']
        sedCommand += ";s/$applyZmumuSkim/%s/g" % getStringRep_bool(option['applyZmumuSkim'])    
        sedCommand += ";s/$applyMuonRadiationFilter/%s/g'" % getStringRep_bool(option['applyMuonRadiationFilter'])    
        sedCommand += " %s > %s" % (cfg_template, cfgFileName)
        runCommand(sedCommand)
        
        # create crab config file
        crabOptions = {
            'datasetpath'       : datasetpath,
            'pset'              : cfgFileName,
            'ui_working_dir'    : os.path.join(submissionDirectory, "crabdir_%s_%s" % (sampleName, label)),
            'user_remote_dir'   : "CMSSW_5_3_x_embed_%s_%s_%s" % (sampleName, label, version),
            'publish_data_name' : "embed_%s_%s_%s" % (sampleName, label, version)
        }
        crabFileName = "crab_embed_%s_%s_cfg.py" % (sampleName, label)
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
shellFile_publish.close()

print("Finished building config files. Now execute 'source %s' to create & submit crab jobs." % shellFileName_create_and_submit)
print("Once all crab jobs have finished processing, execute 'source %s' to publish the samples produced." % shellFileName_publish)
