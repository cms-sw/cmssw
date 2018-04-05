#
# Stand-alone creation of RelVal-plot root file for Strips and Phase0 or Phase1 pixels
# This is derived from Validation/TrackerRecHits/test/
#    SiPixelRecHitsValid_cfg.py
#    SiStripRecHitsValid_cfg.py
# Commented sections support crossing frames with pileup
#  Bill Ford 10 Oct 2017

import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
# process = cms.Process("RecHitsValid", eras.Run2_2016)
process = cms.Process("RecHitsValid", eras.Run2_2017)

process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# (See /Configuration/AlCa/python/autoCond.py)
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# process.load('Configuration.StandardSequences.DigiToRaw_cff')  # for remaking recHits

process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Validation.TrackerRecHits.SiPixelRecHitsValid_cfi")
process.load("Validation.SiPixelPhase1ConfigV.SiPixelPhase1OfflineDQM_harvestingV_cff")
process.load("Validation.TrackerRecHits.SiStripRecHitsValid_cfi")

process.pixRecHitsValid_step = cms.Sequence(process.pixRecHitsValid)
process.stripRecHitsValid_step = cms.Sequence(process.stripRecHitsValid)
process.pixPhase1RecHitsValid_step = cms.Sequence(process.SiPixelPhase1RecHitsAnalyzerV*process.SiPixelPhase1RecHitsHarvesterV)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith(process.pixRecHitsValid_step, process.pixPhase1RecHitsValid_step)
process.validation_step = cms.Sequence(process.pixRecHitsValid_step*process.stripRecHitsValid_step)

process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.workflow = cms.untracked.string('/my/relVal/tracker')

# No pileup
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# From-scratch pileup; needs DigiToRaw in path
# process.load("SimGeneral.MixingModule.mixHighLumPU_cfi")
# process.mix.input.fileNames = cms.untracked.vstring([
#  '/store/relval/CMSSW_7_1_0_pre4/RelValProdMinBias_13/GEN-SIM-RAW/POSTLS171_V1-v2/00000/0275ACA6-3CAA-E311-9DAF-02163E00E62F.root',
#  '/store/relval/CMSSW_7_1_0_pre4/RelValProdMinBias_13/GEN-SIM-RAW/POSTLS171_V1-v2/00000/54DD7A36-3DAA-E311-A92B-0025904B11C0.root',
#  '/store/relval/CMSSW_7_1_0_pre4/RelValProdMinBias_13/GEN-SIM-RAW/POSTLS171_V1-v2/00000/D6560D41-36AA-E311-BF6E-02163E00EBBC.root'
#   ])

# For playback pileup mode
#  TTbar_13/<tier>/PU25ns sample
#  (from config found in DAS for above sample)
#
# process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
# process.mix.input.nbPileupEvents.averageNumber = cms.double(10.000000)
# process.mix.bunchspace = cms.int32(25)
# process.mix.minBunch = cms.int32(-8)  # -8
# process.mix.maxBunch = cms.int32(3)  # 3
# process.mix.input.fileNames = cms.untracked.vstring([
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/0C2DD921-1586-E311-9EA9-02163E00EA98.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/42450CEB-1086-E311-B439-02163E00EB7E.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/502D6922-0986-E311-828F-02163E00A10F.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/7AA4094E-1986-E311-8F3E-003048D2BC06.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/7AF1A921-0D86-E311-86D7-02163E00E6CC.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/88A9162C-0C86-E311-A244-02163E009E82.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/88CDC2A6-1186-E311-A9F5-02163E00E5C7.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/96AC49A8-0F86-E311-BBB4-02163E00EA8B.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/A4846C0D-0B86-E311-8B2E-003048FEB9EE.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/BA98B978-0A86-E311-A451-02163E00E680.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/C04929C2-0D86-E311-A247-003048FEADBC.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/D0E99AF9-0E86-E311-8C6C-00304894528A.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/DCC5CAB6-0786-E311-BBDF-00237DDBEBD0.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/DEACF0AD-0B86-E311-9C27-0030489455E0.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/ECC297E5-1286-E311-846F-001D09F241E6.root',
#   '/store/relval/CMSSW_7_1_0_pre1/RelValMinBias_13/GEN-SIM/POSTLS170_V1-v1/00000/EEC1F7A5-3786-E311-80B3-BCAEC532970F.root'
#
#   ])
# process.mix.playback = True
# process.mix.digitizers = cms.PSet()
# for a in process.aliases: delattr(process, a)

#
# Configure RecHits validator
#
# process.stripRecHitsValid.TH1Resolxrphi.xmax=cms.double(0.00002)
# process.stripRecHitsValid.TH1ResolxStereo.xmax=cms.double(0.01)
# process.stripRecHitsValid.TH1ResolxMatched.xmax=cms.double(0.01)
# process.stripRecHitsValid.TH1ResolyMatched.xmax=cms.double(0.05)
# process.stripRecHitsValid.TH1NumTotrphi.xmax=cms.double(100000.)
# process.stripRecHitsValid.TH1NumTotStereo.xmax=cms.double(100000.)
# process.stripRecHitsValid.TH1NumTotMatched.xmax=cms.double(100000.)
# process.stripRecHitsValid.TH1Numrphi.xmax=cms.double(50000.)
# process.stripRecHitsValid.TH1NumStereo.xmax=cms.double(50000.)
# process.stripRecHitsValid.TH1NumMatched.xmax=cms.double(50000.)

#
# Configure associator
#
# process.pixRecHitsValid.associateHitbySimTrack = cms.bool(True)
# process.stripRecHitsValid.associateHitbySimTrack = cms.bool(True)
#
# Read simHits from prompt collections
process.pixRecHitsValid.ROUList = cms.vstring(
    'TrackerHitsPixelBarrelLowTof', 
    'TrackerHitsPixelBarrelHighTof', 
    'TrackerHitsPixelEndcapLowTof', 
    'TrackerHitsPixelEndcapHighTof'
    )
process.SiPixelPhase1RecHitsAnalyzerV.ROUList = cms.vstring(
    'TrackerHitsPixelBarrelLowTof', 
    'TrackerHitsPixelBarrelHighTof', 
    'TrackerHitsPixelEndcapLowTof', 
    'TrackerHitsPixelEndcapHighTof'
    )
process.stripRecHitsValid.ROUList = cms.vstring(
    'TrackerHitsTIBLowTof', 
    'TrackerHitsTIBHighTof', 
    'TrackerHitsTIDLowTof', 
    'TrackerHitsTIDHighTof', 
    'TrackerHitsTOBLowTof', 
    'TrackerHitsTOBHighTof', 
    'TrackerHitsTECLowTof', 
    'TrackerHitsTECHighTof'
    )

inputfiles=cms.untracked.vstring(
# '/store/relval/CMSSW_9_4_0_pre1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/409345CD-F79C-E711-8383-0CC47A4D76C8.root',
# '/store/relval/CMSSW_9_4_0_pre1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/9C783844-F79C-E711-96E3-0CC47A4D76AA.root',
# '/store/relval/CMSSW_9_4_0_pre1/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/DC94E5C6-F79C-E711-B3EC-0CC47A4D7606.root'

# '/store/relval/CMSSW_9_2_4/RelValSingleMuPt1000_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/2470004C-8D62-E711-817C-0CC47A78A4B8.root',
# '/store/relval/CMSSW_9_2_4/RelValSingleMuPt1000_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/5A844742-8D62-E711-B923-0CC47A4C8E38.root'

# '/store/relval/CMSSW_9_3_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/92X_mcRun2_asymptotic_v2-v1/00000/3219D083-8C63-E711-8FB8-0CC47A4D7628.root',
# '/store/relval/CMSSW_9_3_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/92X_mcRun2_asymptotic_v2-v1/00000/487A39C6-8C63-E711-8FCB-0CC47A4D7698.root',
# '/store/relval/CMSSW_9_3_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/92X_mcRun2_asymptotic_v2-v1/00000/84519579-8D63-E711-86B0-0CC47A4C8EB6.root',
# '/store/relval/CMSSW_9_3_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/92X_mcRun2_asymptotic_v2-v1/00000/9EA8737B-8C63-E711-A46C-0025905B85CC.root',
# '/store/relval/CMSSW_9_3_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/92X_mcRun2_asymptotic_v2-v1/00000/F6CF72E4-8B63-E711-96DA-0025905A6068.root',
# '/store/relval/CMSSW_9_3_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW-HLTDEBUG/92X_mcRun2_asymptotic_v2-v1/00000/FC762B97-8C63-E711-8CEE-0CC47A7C353E.root'

'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/00D622D4-A79C-E711-8A41-0CC47A7C35D8.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/28D592D3-A79C-E711-BE47-0CC47A7C3458.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/2C0B8FD4-A79C-E711-BF8F-0CC47A4D761A.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/54C1ABAB-A89C-E711-BEDF-0025905A60BE.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/5C3C3FD7-A79C-E711-91A6-0CC47A7C35A8.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/7E1D18E2-A79C-E711-969D-0025905A610A.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/E2715BD7-A79C-E711-807F-0025905B859E.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/E2772BAB-A89C-E711-945D-0025905B857C.root',
'/store/relval/CMSSW_9_4_0_pre1/RelValQCD_Pt_3000_3500_13/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v3-v1/00000/FE788BD4-A79C-E711-B5D3-0CC47A4C8F08.root'

)
secinputfiles=cms.untracked.vstring(
)

process.source = cms.Source("PoolSource",
    fileNames = inputfiles,
    secondaryFileNames = secinputfiles
    , inputCommands=cms.untracked.vstring('keep *', 
      'drop l1tEMTFHitExtras_simEmtfDigis_CSC_HLT',
      'drop l1tEMTFHitExtras_simEmtfDigis_RPC_HLT',
      'drop l1tEMTFTrackExtras_simEmtfDigis__HLT'
    )
)

# # Input source
# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring('file:../10008.0_SingleMuPt100+SingleMuPt100_pythia8_2017_GenSimFullINPUT+DigiFull_2017+RecoFull_2017+ALCAFull_2017+HARVESTFull_2017/step2.root'),
#     # fileNames = cms.untracked.vstring('file:../1321.0_SingleMuPt100_UP15+SingleMuPt100_UP15INPUT+DIGIUP15+RECOUP15+HARVESTUP15/step2.root'),
#     secondaryFileNames = cms.untracked.vstring()
# )

process.options = cms.untracked.PSet(
  # SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Insert this in path to see what products the event contains
process.content = cms.EDAnalyzer("EventContentAnalyzer")

# To enable debugging:
# [scram b clean ;] scram b USER_CXXFLAGS="-DEDM_ML_DEBUG"

# process.load("SimTracker.TrackerHitAssociation.test.messageLoggerDebug_cff")

process.MessageLogger.cerr.FwkReport.reportEvery = 1

# Number of events (-1 = all)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.p1 = cms.Path(
    # process.content*
    process.mix
    # *process.DigiToRaw  # for remaking recHits
    *process.RawToDigi
    *process.L1Reco
    *process.reconstruction
    *process.validation_step
    *process.dqmSaver
    )

# # customisation of the process.

# # Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
# from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

# #call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
# process = setCrossingFrameOn(process)

# # End of customisation functions

