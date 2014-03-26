import FWCore.ParameterSet.Config as cms

process = cms.Process("siStripRecHitsValid")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START71_V1::All'

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Validation.TrackerRecHits.SiStripRecHitsValid_cfi")
process.stripRecHitsValid.OutputMEsInRootFile = cms.bool(True)
# process.stripRecHitsValid.outputFile="sistriprechitshisto.root"
process.stripRecHitsValid.TH1Resolxrphi.xmax=cms.double(0.00002)
process.stripRecHitsValid.TH1ResolxStereo.xmax=cms.double(0.01)
process.stripRecHitsValid.TH1ResolxMatched.xmax=cms.double(0.01)
process.stripRecHitsValid.TH1ResolyMatched.xmax=cms.double(0.05)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1

inputfiles=cms.untracked.vstring(
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/2C41AD32-AEA1-E311-A53F-0025904B26A8.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/50F23E79-ABA1-E311-B30C-02163E00EAF1.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/6E0B8D4D-B1A1-E311-8DCE-002590494E94.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/74219F0C-BAA1-E311-823D-02163E00CC88.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/8A78266D-B5A1-E311-9D17-02163E008D9B.root'
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v2/00000/14654820-61AA-E311-BF2B-02163E00E7E3.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v2/00000/702A1D67-47AA-E311-8700-02163E00EA83.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v2/00000/8EC99383-4CAA-E311-B082-02163E00E932.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v2/00000/92400642-55AA-E311-9973-02163E00E8F7.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-RECO/POSTLS171_V1-v2/00000/E2BFB2C0-41AA-E311-BDDA-00304894529A.root'
)
secinputfiles=cms.untracked.vstring(
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/14C6EDEE-9EA1-E311-AF11-02163E00E7BE.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/1C276DB4-A5A1-E311-BC9C-02163E007A03.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/22459DB0-A5A1-E311-8F2C-02163E009E2B.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/2463C821-ADA1-E311-B609-02163E00A18A.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/4266E8D2-A1A1-E311-8589-02163E00E92E.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/4EFE65B9-9EA1-E311-8593-02163E008DAD.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/96E0FB42-A0A1-E311-91DF-02163E00EA58.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/981368FD-9EA1-E311-8155-002590494F70.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/A28039F1-9EA1-E311-8315-02163E00E964.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/C615CC01-A2A1-E311-A35A-02163E008D65.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/D0A7A567-9EA1-E311-81B4-02163E009DD0.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/D6383DF2-9EA1-E311-9D2E-02163E00E84C.root',
#  '/store/relval/CMSSW_7_1_0_pre3/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v1/00000/DCD5EEA2-9EA1-E311-A14B-02163E00EA4A.root'
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/00941262-26AA-E311-BBCD-02163E00E8AB.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/0E52B8B7-31AA-E311-9C67-02163E00EB0C.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/108ACEE9-1DAA-E311-8ABE-02163E00EA2F.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/164BE658-2AAA-E311-A673-02163E00EA48.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/26189FDD-1EAA-E311-A33B-02163E00EAC2.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/4C48E4FC-1DAA-E311-AD21-02163E00EA9D.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/86FBA898-20AA-E311-ABBE-02163E00E5ED.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/9CAFC9E2-1DAA-E311-A329-02163E00EB23.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/9CF68113-25AA-E311-916C-02163E008DA4.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/BC3AD83B-1FAA-E311-842C-02163E00EB4F.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/C0DEE0FF-36AA-E311-B3A2-02163E00EA43.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/E07C78C9-21AA-E311-A7DC-02163E00E857.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/F4E8FD65-2AAA-E311-895C-02163E00EAFA.root',
 '/store/relval/CMSSW_7_1_0_pre4/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/POSTLS171_V1-v2/00000/F6D97D24-1CAA-E311-82C7-02163E00E7B0.root'
)
process.source = cms.Source("PoolSource",
    fileNames = inputfiles,
    secondaryFileNames = secinputfiles
)

process.p1 = cms.Path(
    process.mix
    *process.siStripMatchedRecHits
    *process.stripRecHitsValid
    )

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

