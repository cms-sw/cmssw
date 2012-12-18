#!/usr/bin/env python

import os
import shlex
import string
import subprocess

samples = {
##     'simDYtoTauTau_mutau' : {
##         'datasetpath'      : '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/Summer12_DR53X-PU_S10_START53_V7A-v1/AODSIM',
##         'dbs_url'          : 'http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet',
##         'events_processed' : 30459503,
##         'events_per_job'   : 10000,
##         'type'             : 'MC',
##         'channel'          : 'mutau',
##         'srcWeights'       : [],
##         'srcGenFilterInfo' : ''
##     },
##     'simDYtoMuMu_noEvtSel_embedEqRH_cleanEqDEDX_replaceGenMuons_by_mutau_embedAngleEq90' : {
##         'datasetpath'      : '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/aburgmei-Summer12_DYJetsToLL_DR53X_PU_S10_START53_V7A_v2_GENEmbed_embedded_trans1_tau116_v2-5ef1c0fd428eb740081f19333520fdc8/USER',
##         'dbs_url'          : 'http://cmsdbsprod.cern.ch/cms_dbs_ph_analysis_01/servlet/DBSServlet',
##         'events_processed' : -1,
##         'events_per_job'   : 10000,
##         'type'             : 'MC',
##         'channel'          : 'mutau',
##         'srcWeights'       : [],
##         'srcGenFilterInfo' : 'generator:minVisPtFilter'
##     },
##     'simDYtoMuMu_noEvtSel_embedEqRH_cleanEqDEDX_replaceRECMuons_by_mutau_embedAngleEq90' : {
##         'datasetpath'      : '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/aburgmei-Summer12_DYJetsToLL_DR53X_PU_S10_START53_V7A_v2_RECEmbed_embedded_trans1_tau116_v2-5ef1c0fd428eb740081f19333520fdc8/USER',
##         'dbs_url'          : 'http://cmsdbsprod.cern.ch/cms_dbs_ph_analysis_01/servlet/DBSServlet',
##         'events_processed' : -1,
##         'events_per_job'   : 10000,
##         'type'             : 'MC',
##         'channel'          : 'mutau',
##         'srcWeights'       : [],
##         'srcGenFilterInfo' : 'generator:minVisPtFilter'
##     },
    'simDYtoMuMu_noEvtSel_embedEqPF_replaceGenMuons_by_mutau_embedAngleEq0_noVisPtCuts' : {
        'datasetpath'      : '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/aburgmei-Summer12_DYJetsToLL_DR53X_PU_S10_START53_V7A_v2_PFGENEmbed_Angle0_VisPt0_embedded_trans1_tau116_v2-5ef1c0fd428eb740081f19333520fdc8/USER',
        'dbs_url'          : 'http://cmsdbsprod.cern.ch/cms_dbs_ph_analysis_01/servlet/DBSServlet',
        'events_processed' : -1,
        'events_per_job'   : 10000,
        'type'             : 'MC',
        'channel'          : 'mutau',
        'srcWeights'       : [],
        'srcGenFilterInfo' : ''
    },
    'simDYtoMuMu_noEvtSel_embedEqPF_replaceGenMuons_by_mutau_embedAngleEq90_noVisPtCuts' : {
        'datasetpath'      : '/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/aburgmei-Summer12_DYJetsToLL_DR53X_PU_S10_START53_V7A_v2_RHGENEmbed_Angle90_VisPt0_embedded_trans1_tau116_v2-5ef1c0fd428eb740081f19333520fdc8/USER',
        'dbs_url'          : 'http://cmsdbsprod.cern.ch/cms_dbs_ph_analysis_01/servlet/DBSServlet',
        'events_processed' : -1,
        'events_per_job'   : 10000,
        'type'             : 'MC',
        'channel'          : 'mutau',
        'srcWeights'       : [],
        'srcGenFilterInfo' : ''
    },
}

version = "v1_5_1"

crab_template = string.Template('''
[CRAB]
jobtype = cmssw
scheduler = glidein
use_server = 1

[CMSSW]
datasetpath = $datasetpath
dbs_url = $dbs_url
pset = $pset
output_file = validateMCEmbedding_plots.root
# CV: use for MC
total_number_of_events = -1
events_per_job = $events_per_job
# CV: use for Data
#lumi_mask = /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions12/8TeV/DCSOnly/json_DCSONLY.txt
#total_number_of_lumis = 2
#number_of_jobs = 1
#runselection = 190450-190790

[USER]
ui_working_dir = $ui_working_dir
return_data = 0
copy_data = 1
publish_data = 0
storage_element = T2_CH_CERN
user_remote_dir = $user_remote_dir
''')

cfg_template = "validateMCEmbedding_cfg.py"

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
    # create config file for cmsRun
    cfgFileName = "validateMCEmbedding_%s_cfg.py" % sampleName
    runCommand('rm -f %s' % cfgFileName)
    sedCommand  = "sed 's/#__//g"
    isMC = None
    if sampleOption['type'] == "Data":
        isMC = False
    elif sampleOption['type'] == "MC":
        isMC = True
    else:
        raise ValueError ("Sample = %s is of invalid type = %s !!" % (sampleName, sampleOption['type']))
    sedCommand += ";s/$isMC/%s/g" % getStringRep_bool(isMC)
    sedCommand += ";s/$channel/%s/g" % sampleOption['channel']
    srcWeights_string = "[ "
    for srcWeight in sampleOption['srcWeights']:
        if len(srcWeights_string) > 0:
            srcWeights_string += ", "
        srcWeights_string += "'%s'" % srcWeight
    srcWeights_string += " ]"
    sedCommand += ";s/$srcWeights/%s/g" % srcWeights_string
    sedCommand += ";s/$srcGenFilterInfo/%s/g" % sampleOption['srcGenFilterInfo']
    sedCommand += "'"
    sedCommand += " %s > %s" % (cfg_template, cfgFileName)
    runCommand(sedCommand)
        
    # create crab config file
    crabOptions = {
        'datasetpath'       : sampleOption['datasetpath'],
        'dbs_url'           : sampleOption['dbs_url'],
        'events_per_job'    : sampleOption['events_per_job'],
        'pset'              : cfgFileName,
        'ui_working_dir'    : os.path.join(submissionDirectory, "crabdir_%s" % sampleName),
        'user_remote_dir'   : "CMSSW_5_3_x/plots/EmbeddingValidation/%s/%s" % (version, sampleName)
    }
    crabFileName = "crab_validateMCEmbedding_%s_cfg.py" % sampleName
    crabFileName_full = os.path.join(submissionDirectory, crabFileName)
    crabFile = open(crabFileName_full, 'w')
    crabConfig = crab_template.substitute(crabOptions)
    crabFile.write(crabConfig)
    crabFile.close()

    # keep track of commands necessary to create, submit and publish crab jobs
    crabCommands_create_and_submit.append('crab -create -cfg %s' % crabFileName_full)
    if (sampleOption['events_processed'] / sampleOption['events_per_job']) < 1000:
        crabCommands_create_and_submit.append('crab -submit -c %s' % crabOptions['ui_working_dir'])
    else:
        numJobs = (sampleOption['events_processed'] / sampleOption['events_per_job'])
        if (sampleOption['events_processed'] % sampleOption['events_per_job']) != 0:
            numJobs = numJobs + 1
        numJobs_per_submitCall = 1000
        numSubmitCalls = (numJobs / numJobs_per_submitCall)
        if (numJobs % numJobs_per_submitCall) != 0:
            numSubmitCalls = numSubmitCalls + 1
        for submitIdx in range(numSubmitCalls):
            jobId_first = submitIdx*1000 + 1
            jobId_last  = (submitIdx + 1)*1000
            if jobId_last > numJobs:
                jobId_last = numJobs
            crabCommands_create_and_submit.append('crab -submit %i-%i -c %s' % (jobId_first, jobId_last, crabOptions['ui_working_dir']))
    
shellFileName_create_and_submit = "validateMCEmbedding_crab_create_and_submit.sh"
shellFile_create_and_submit = open(shellFileName_create_and_submit, "w")
for crabCommand in crabCommands_create_and_submit:
    shellFile_create_and_submit.write("%s\n" % crabCommand)
shellFile_create_and_submit.close()

print("Finished building config files. Now execute 'source %s' to create & submit crab jobs." % shellFileName_create_and_submit)

