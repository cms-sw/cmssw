import sys, os
import FWCore.ParameterSet.Config as cms

isSignal = False
isBckg = True
isData = False
isSkimmedSample = False
GTAG = 'POSTLS172_V3::All'

#debug input files 
#this list is overwritten by CRAB
InputFileList = cms.untracked.vstring(
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/2C7EF234-5E21-E411-99D8-0025905A60C6.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/38D93E91-6221-E411-ADF4-0025905A60F4.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/66520EC6-5C21-E411-9A49-0025905A48F2.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/6E65F2CA-6221-E411-88B0-0025905B85F6.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/A0FE331B-5B21-E411-AE0C-0025905A48D6.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/AC325A29-5F21-E411-87C5-0025905B8576.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/B2185429-A121-E411-B6B1-0025905A6088.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/B40BDDCF-9F21-E411-8926-0025905B85E8.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/B467B1A5-5D21-E411-AA34-0025905A611E.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/B4E1729F-6421-E411-B76E-0025905A6122.root',
   '/store/relval/CMSSW_7_2_0_pre3/RelValZmumuJets_Pt_20_300_13/GEN-SIM-RECO/PU25ns_POSTLS172_V3_CondDBv2-v1/00000/F2B25D76-5C21-E411-973A-0025905A6136.root',
)

#main EDM tuple cfg that depends on the above parameters
execfile( os.path.expandvars('${CMSSW_BASE}/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/HSCParticleProducer_cfg.py') )
