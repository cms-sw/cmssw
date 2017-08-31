#! /usr/bin/env python

import os
import time
import sys
import re
import random
from threading import Thread

scriptPath = os.path.dirname( os.path.abspath(sys.argv[0]) )
if scriptPath not in sys.path:
    sys.path.append(scriptPath)

        
class testit(Thread):
    def __init__(self,dirName, commandList):
        Thread.__init__(self)
        self.dirName = dirName
        self.commandList = commandList
        self.status=-1
        self.report=''
        self.nfail=[]
        self.npass=[]

        return
    
    def run(self):

        startime='date %s' %time.asctime()
        exitCodes = []

        for command in self.commandList:

            if not os.path.exists(self.dirName):
                os.makedirs(self.dirName)

            commandbase = command.replace(' ','_').replace('/','_')
            logfile='%s.log' % commandbase[:150].replace("'",'').replace('"','').replace('../','')
            
            executable = 'cd '+self.dirName+'; '+command+' > '+logfile+' 2>&1'

            ret = os.system(executable)
            exitCodes.append( ret )

        endtime='date %s' %time.asctime()
        tottime='%s-%s'%(endtime,startime)
    
        for i in range(len(self.commandList)):
            command = self.commandList[i]
            exitcode = exitCodes[i]
            if exitcode != 0:
                log='%s : FAILED - time: %s s - exit: %s\n' %(command,tottime,exitcode)
                self.report+='%s\n'%log
                self.nfail.append(1)
                self.npass.append(0)
            else:
                log='%s : PASSED - time: %s s - exit: %s\n' %(command,tottime,exitcode)
                self.report+='%s\n'%log
                self.nfail.append(0)
                self.npass.append(1)

        return

class StandardTester(object):

    def __init__(self, nThrMax=4):

        self.threadList = []
        self.maxThreads = nThrMax
        self.prepare()

        return

    def activeThreads(self):

        nActive = 0
        for t in self.threadList:
            if t.isAlive() : nActive += 1

        return nActive

    def prepare(self):
    
        self.devPath = os.environ['LOCALRT'] + '/src/'
        self.relPath = self.devPath
        if 'CMSSW_RELEASE_BASE' in os.environ and (os.environ['CMSSW_RELEASE_BASE'] != ""): self.relPath = os.environ['CMSSW_RELEASE_BASE'] + '/src/'

        lines = { 'read312RV' : ['cmsRun '+self.file2Path('Utilities/ReleaseScripts/scripts/read312RV_cfg.py')], 
                  'fastsim'   : ["cmsDriver.py TTbar_8TeV_TuneCUETP8M1_cfi  --conditions auto:run1_mc --fast  -n 100 --eventcontent AODSIM,DQM --relval 100000,1000 -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,EI,VALIDATION  --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --datatier GEN-SIM-DIGI-RECO,DQMIO --beamspot Realistic8TeVCollision"],
                  'fastsim1'  : ["cmsDriver.py TTbar_13TeV_TuneCUETP8M1_cfi --conditions auto:run2_mc --fast  -n 100 --eventcontent AODSIM,DQM --relval 100000,1000 -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,EI,VALIDATION  --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --datatier GEN-SIM-DIGI-RECO,DQMIO --beamspot NominalCollision2015 --era Run2_25ns"],
                  'fastsim2'  : ["cmsDriver.py TTbar_13TeV_TuneCUETP8M1_cfi --conditions auto:run2_mc --fast  -n 100 --eventcontent AODSIM,DQM --relval 100000,1000 -s GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,EI,VALIDATION  --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --datatier GEN-SIM-DIGI-RECO,DQMIO --beamspot NominalCollision2015 --era Run2_2016"],
                  'pat1'      : ['cmsRun '+self.file2Path('PhysicsTools/PatAlgos/test/IntegrationTest_cfg.py')],
                }

        hltFlag_data = ' realData=True  globalTag=@ inputFiles=@ '
        hltFlag_mc   = ' realData=False globalTag=@ inputFiles=@ '
 
        hltTests = {
                    'hlt_mc_Fake' : ['cmsDriver.py TTbar_Tauola_8TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW  --mc --scenario=pp -n 10 --conditions auto:run1_mc_Fake --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --fileout file:RelVal_Raw_Fake_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_Fake.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:Fake,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run1_mc_Fake --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_Fake_MC.root --fileout file:RelVal_Raw_Fake_MC_HLT_RECO.root'], 
                    'hlt_mc_Fake1': ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_Fake1 --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_25ns --fileout file:RelVal_Raw_Fake1_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_Fake1.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:Fake1,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_Fake1 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_25ns --processName=HLTRECO --filein file:RelVal_Raw_Fake1_MC.root --fileout file:RelVal_Raw_Fake1_MC_HLT_RECO.root'], 
                    'hlt_mc_Fake2': ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_Fake2 --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2016 --fileout file:RelVal_Raw_Fake2_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_Fake2.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:Fake2,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_Fake2 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2016 --processName=HLTRECO --filein file:RelVal_Raw_Fake2_MC.root --fileout file:RelVal_Raw_Fake2_MC_HLT_RECO.root'], 
                    'hlt_mc_2e34v21' : ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_2e34v21 --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_2e34v21_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_2e34v21.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:2e34v21,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_2e34v21 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_2e34v21_MC.root --fileout file:RelVal_Raw_2e34v21_MC_HLT_RECO.root'], 
                    'hlt_mc_2e34v22' : ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_2e34v22 --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_2e34v22_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_2e34v22.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:2e34v22,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_2e34v22 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_2e34v22_MC.root --fileout file:RelVal_Raw_2e34v22_MC_HLT_RECO.root'], 
                    'hlt_mc_2e34v30' : ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_2e34v30 --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_2e34v30_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_2e34v30.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:2e34v30,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_2e34v30 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_2e34v30_MC.root --fileout file:RelVal_Raw_2e34v30_MC_HLT_RECO.root'], 
                    'hlt_mc_GRun' : ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_GRun --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_GRun_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_GRun.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:GRun,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_GRun --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_GRun_MC.root --fileout file:RelVal_Raw_GRun_MC_HLT_RECO.root'], 
                    'hlt_mc_HIon' : ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=HeavyIons -n 10 --conditions auto:run2_mc_HIon --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2016,Run2_HI --fileout file:RelVal_Raw_HIon_MC.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_HIon.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:HIon,RAW2DIGI,L1Reco,RECO --mc --scenario=HeavyIons -n 10 --conditions auto:run2_mc_HIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2016,Run2_HI --processName=HLTRECO --filein file:RelVal_Raw_HIon_MC.root --fileout file:RelVal_Raw_HIon_MC_HLT_RECO.root'],
                    'hlt_mc_PIon' : ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_PIon --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_PIon_MC.root', 
                               'cmsRun ' + self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_PIon.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:PIon,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_PIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_PIon_MC.root --fileout file:RelVal_Raw_PIon_MC_HLT_RECO.root'],
                    'hlt_mc_PRef' : ['cmsDriver.py TTbar_Tauola_13TeV_TuneCUETP8M1_cfi -s GEN,SIM,DIGI,L1,DIGI2RAW --mc --scenario=pp -n 10 --conditions auto:run2_mc_PRef --relval 9000,50 --datatier "GEN-SIM-RAW" --eventcontent RAWSIM --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_PRef_MC.root', 
                               'cmsRun ' + self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_PRef.py')+hltFlag_mc,
                               'cmsDriver.py RelVal -s HLT:PRef,RAW2DIGI,L1Reco,RECO --mc --scenario=pp -n 10 --conditions auto:run2_mc_PRef --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_PRef_MC.root --fileout file:RelVal_Raw_PRef_MC_HLT_RECO.root'],
                    'hlt_data_Fake' : ['cmsDriver.py RelVal -s L1REPACK:GT1   --data --scenario=pp -n 10 --conditions auto:run1_hlt_Fake --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --fileout file:RelVal_Raw_Fake_DATA.root --filein /store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_Fake.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:Fake,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run1_data_Fake --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --processName=HLTRECO --filein file:RelVal_Raw_Fake_DATA.root --fileout file:RelVal_Raw_Fake_DATA_HLT_RECO.root'],
                    'hlt_data_Fake1': ['cmsDriver.py RelVal -s L1REPACK:GCTGT --data --scenario=pp -n 10 --conditions auto:run2_hlt_Fake1 --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_25ns --fileout file:RelVal_Raw_Fake1_DATA.root --filein /store/data/Run2015D/MuonEG/RAW/v1/000/256/677/00000/80950A90-745D-E511-92FD-02163E011C5D.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_Fake1.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:Fake1,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_Fake1 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_25ns --processName=HLTRECO --filein file:RelVal_Raw_Fake1_DATA.root --fileout file:RelVal_Raw_Fake1_DATA_HLT_RECO.root'],
                    'hlt_data_Fake2': ['cmsDriver.py RelVal -s L1REPACK:Full --data --scenario=pp -n 10 --conditions auto:run2_hlt_Fake2 --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2016 --fileout file:RelVal_Raw_Fake2_DATA.root --filein /store/data/Run2015D/MuonEG/RAW/v1/000/256/677/00000/80950A90-745D-E511-92FD-02163E011C5D.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_Fake2.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:Fake2,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_Fake2 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2016 --processName=HLTRECO --filein file:RelVal_Raw_Fake2_DATA.root --fileout file:RelVal_Raw_Fake2_DATA_HLT_RECO.root'],
                    'hlt_data_2e34v21' : ['cmsDriver.py RelVal -s L1REPACK:Full --data --scenario=pp -n 10 --conditions auto:run2_hlt_2e34v21 --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_2e34v21_DATA.root --filein /store/data/Run2017A/HLTPhysics4/RAW/v1/000/295/606/00000/36DE5E0A-3645-E711-8FA1-02163E01A43B.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_2e34v21.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:2e34v21,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_2e34v21 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_2e34v21_DATA.root --fileout file:RelVal_Raw_2e34v21_DATA_HLT_RECO.root'],
                    'hlt_data_2e34v22' : ['cmsDriver.py RelVal -s L1REPACK:Full --data --scenario=pp -n 10 --conditions auto:run2_hlt_2e34v22 --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_2e34v22_DATA.root --filein /store/data/Run2017A/HLTPhysics4/RAW/v1/000/295/606/00000/36DE5E0A-3645-E711-8FA1-02163E01A43B.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_2e34v22.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:2e34v22,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_2e34v22 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_2e34v22_DATA.root --fileout file:RelVal_Raw_2e34v22_DATA_HLT_RECO.root'],
                    'hlt_data_2e34v30' : ['cmsDriver.py RelVal -s L1REPACK:Full --data --scenario=pp -n 10 --conditions auto:run2_hlt_2e34v30 --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_2e34v30_DATA.root --filein /store/data/Run2017A/HLTPhysics4/RAW/v1/000/295/606/00000/36DE5E0A-3645-E711-8FA1-02163E01A43B.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_2e34v30.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:2e34v30,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_2e34v30 --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_2e34v30_DATA.root --fileout file:RelVal_Raw_2e34v30_DATA_HLT_RECO.root'],
                    'hlt_data_GRun' : ['cmsDriver.py RelVal -s L1REPACK:Full --data --scenario=pp -n 10 --conditions auto:run2_hlt_GRun --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --fileout file:RelVal_Raw_GRun_DATA.root --filein /store/data/Run2017A/HLTPhysics4/RAW/v1/000/295/606/00000/36DE5E0A-3645-E711-8FA1-02163E01A43B.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_GRun.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:GRun,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_GRun --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_GRun_DATA.root --fileout file:RelVal_Raw_GRun_DATA_HLT_RECO.root'],
                    'hlt_data_HIon' : ['cmsDriver.py RelVal -s L1REPACK:Full2015Data --data --scenario=HeavyIons -n 10 --conditions auto:run2_hlt_HIon --relval 9000,50 --datatier "RAW" --eventcontent RAW --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2016,Run2_HI --fileout file:RelVal_Raw_HIon_DATA.root --filein /store/hidata/HIRun2015/HIHardProbes/RAW-RECO/HighPtJet-PromptReco-v1/000/263/689/00000/1802CD9A-DDB8-E511-9CF9-02163E0138CA.root',
                               'cmsRun '+self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_HIon.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:HIon,RAW2DIGI,L1Reco,RECO --data --scenario=HeavyIons -n 10 --conditions auto:run2_data_HIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2016,Run2_HI --processName=HLTRECO --filein file:RelVal_Raw_HIon_DATA.root --fileout file:RelVal_Raw_HIon_DATA_HLT_RECO.root'],
                    'hlt_data_PIon' : ['cmsDriver.py RelVal -s L1REPACK:Full --data --scenario=pp -n 10 --conditions auto:run2_hlt_PIon --relval 9000,50 --datatier "RAW" --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --eventcontent RAW --fileout file:RelVal_Raw_PIon_DATA.root --filein /store/data/Run2017A/HLTPhysics4/RAW/v1/000/295/606/00000/36DE5E0A-3645-E711-8FA1-02163E01A43B.root',
                               'cmsRun ' + self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_PIon.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:PIon,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_PIon --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_PIon_DATA.root --fileout file:RelVal_Raw_PIon_DATA_HLT_RECO.root'],
                    'hlt_data_PRef' : ['cmsDriver.py RelVal -s L1REPACK:Full --data --scenario=pp -n 10 --conditions auto:run2_hlt_PRef --relval 9000,50 --datatier "RAW" --customise=HLTrigger/Configuration/CustomConfigs.L1T --era Run2_2017 --eventcontent RAW --fileout file:RelVal_Raw_PRef_DATA.root --filein /store/data/Run2017A/HLTPhysics4/RAW/v1/000/295/606/00000/36DE5E0A-3645-E711-8FA1-02163E01A43B.root',
                               'cmsRun ' + self.file2Path('HLTrigger/Configuration/test/OnLine_HLT_PRef.py')+hltFlag_data,
                               'cmsDriver.py RelVal -s HLT:PRef,RAW2DIGI,L1Reco,RECO --data --scenario=pp -n 10 --conditions auto:run2_data_PRef --relval 9000,50 --datatier "RAW-HLT-RECO" --eventcontent FEVTDEBUGHLT --customise=HLTrigger/Configuration/CustomConfigs.L1THLT --era Run2_2017 --processName=HLTRECO --filein file:RelVal_Raw_PRef_DATA.root --fileout file:RelVal_Raw_PRef_DATA_HLT_RECO.root'],
                     }

        self.commands={}
        for dirName, command in lines.items():
            self.commands[dirName] = command

        for dirName, commandList in hltTests.items():
            self.commands[dirName] = commandList
        return
	
    def dumpTest(self):
        print ",".join(self.commands.keys())
        return

    def file2Path(self,rFile):

        fullPath = self.relPath + rFile
        if os.path.exists(self.devPath + rFile): fullPath = self.devPath + rFile
        return fullPath

    def runTests(self, testList = None):

        actDir = os.getcwd()

        if not os.path.exists('addOnTests'):
            os.makedirs('addOnTests')
        os.chdir('addOnTests')

        nfail=0
    	npass=0
    	report=''
    	
    	print 'Running in %s thread(s)' % self.maxThreads
    	
        for dirName, command in self.commands.items():

    	    if testList and not dirName in testList:
                del self.commands[dirName]
                continue

            # make sure we don't run more than the allowed number of threads:
    	    while self.activeThreads() >= self.maxThreads:
    	        time.sleep(10)
                continue
    	    
    	    print 'Preparing to run %s' % str(command)
    	    current = testit(dirName, command)
    	    self.threadList.append(current)
    	    current.start()
            time.sleep(random.randint(1,5)) # try to avoid race cond by sleeping random amount of time [1,5] sec 
            
        # wait until all threads are finished
        while self.activeThreads() > 0:
    	    time.sleep(5)
    	    
    	# all threads are done now, check status ...
    	for pingle in self.threadList:
    	    pingle.join()
            for f in pingle.nfail: nfail  += f
            for p in pingle.npass: npass  += p
    	    report += pingle.report
    	    print pingle.report
            sys.stdout.flush()
            
    	reportSumm = '\n %s tests passed, %s failed \n' %(npass,nfail)
    	print reportSumm
    	
    	runall_report_name='runall-report.log'
    	runall_report=open(runall_report_name,'w')
    	runall_report.write(report+reportSumm)
    	runall_report.close()

        # get the logs to the logs dir:
        print '==> in :', os.getcwd()
        print '    going to copy log files to logs dir ...'
        if not os.path.exists('logs'):
            os.makedirs('logs')
        for dirName in self.commands:
            cmd = "for L in `ls "+dirName+"/*.log`; do cp $L logs/cmsDriver-`dirname $L`_`basename $L` ; done"
            print "going to ",cmd
            os.system(cmd)

        import pickle
        pickle.dump(self.commands, open('logs/addOnTests.pkl', 'w') )

        os.chdir(actDir)
        
    	return

    def upload(self, tgtDir):

        print "in ", os.getcwd()

        if not os.path.exists(tgtDir):
            os.makedirs(tgtDir)
        
        cmd = 'tar cf - addOnTests.log addOnTests/logs | (cd '+tgtDir+' ; tar xf - ) '
        try:
            print 'executing: ',cmd
            ret = os.system(cmd)
            if ret != 0:
                print "ERROR uploading logs:", ret, cmd
        except Exception as e:
            print "EXCEPTION while uploading addOnTest-logs : ", str(e)
            
    	return

                
def main(argv) :

    import getopt
    
    try:
        opts, args = getopt.getopt(argv, "dj:t:", ["nproc=", 'uploadDir=', 'tests=','noRun','dump'])
    except getopt.GetoptError as e:
        print "unknown option", str(e)
        sys.exit(2)
        
    np        = 4
    uploadDir = None
    runTests  = True
    testList  = None
    dump      = False
    for opt, arg in opts :
        if opt in ('-j', "--nproc" ):
            np=int(arg)
        if opt in ("--uploadDir", ):
            uploadDir = arg
        if opt in ('--noRun', ):
            runTests = False
        if opt in ('-d','--dump', ):
            dump = True
        if opt in ('-t','--tests', ):
            testList = arg.split(",")

    tester = StandardTester(np)
    if dump:
        tester.dumpTest()
    else:
        if runTests:
            tester.runTests(testList)
        if uploadDir:
            tester.upload(uploadDir)
    return
    
if __name__ == '__main__' :
    main(sys.argv[1:])
