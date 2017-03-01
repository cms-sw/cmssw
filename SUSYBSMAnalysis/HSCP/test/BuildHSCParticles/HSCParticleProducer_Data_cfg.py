import sys, os
import FWCore.ParameterSet.Config as cms

isSignal = False
isBckg = False
isData = True
isSkimmedSample = False
GTAG = 'GR_R_72_V2::All'

#debug input files 
#this list is overwritten by CRAB
InputFileList = cms.untracked.vstring(
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/00346527-C91C-E411-AB5E-02163E00ECEF.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/00ADAB1A-BC1C-E411-8EF1-002590494C40.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0200D6AF-D41C-E411-8420-0025904B0FC0.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0268317F-AB1C-E411-8022-02163E00ECFB.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/02F71E4D-B61C-E411-888D-02163E00CFB4.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/02FCA65D-AF1C-E411-BED1-18A90555637A.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0405F2E2-B01C-E411-BC20-02163E00E5B2.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/041A1E18-C21C-E411-A77A-00259029EF3E.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/044F878E-E91C-E411-B9D8-02163E00CAA2.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/04C9C6B8-D31C-E411-88E9-003048F0E7BE.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0634987C-A91C-E411-A9B5-02163E009C1E.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0637E986-BE1C-E411-ACFF-02163E00EF94.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0644EC20-C91C-E411-A376-02163E008EEA.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/068E6015-CB1C-E411-96C8-02163E00E95C.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/06D99999-BD1C-E411-896E-0025B3203748.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/08361980-BC1C-E411-97CB-003048C9C1D4.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0836D993-B61C-E411-B245-02163E009BA7.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/08A1A887-C41C-E411-9BCB-02163E00FEC3.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/08B88B4E-D11C-E411-9EDB-02163E00B7A3.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/08CE58CE-EB1C-E411-BABA-02163E010110.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/08D1A881-9F1C-E411-8E09-02163E00ECE6.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/08F06953-BB1C-E411-8F27-003048C9C1D0.root',
   '/store/relval/CMSSW_7_2_0_pre3/SingleMu/RECO/GR_R_72_V2_frozenHLT_RelVal_mu2012D-v1/00000/0A44FE81-A71C-E411-862A-02163E0104C0.root',
)

#main EDM tuple cfg that depends on the above parameters
execfile( os.path.expandvars('${CMSSW_BASE}/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/HSCParticleProducer_cfg.py') )
