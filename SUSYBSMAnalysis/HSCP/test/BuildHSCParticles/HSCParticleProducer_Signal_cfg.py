import sys, os
import FWCore.ParameterSet.Config as cms

isSignal = True
isBckg = False
isData = False
isSkimmedSample = False
GTAG = 'START72_V1::All'
InputFileList = cms.untracked.vstring()

#debug input files 
#this list is overwritten by CRAB
for i in range(0,10):
   InputFileList.extend(["file:" + os.path.expandvars("${CMSSW_BASE}/src/") + "SampleProd/FARM_RECO/outputs/gluino1TeV_RECO_%04i.root" % i])

#main EDM tuple cfg that depends on the above parameters
execfile( os.path.expandvars('${CMSSW_BASE}/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/HSCParticleProducer_cfg.py') )
