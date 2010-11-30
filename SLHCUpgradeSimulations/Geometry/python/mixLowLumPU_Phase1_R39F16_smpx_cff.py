# Phase 1 R39v26 minbias pileup files
# E34 cm-2s-1
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in the design LHC (10**34)
from SimGeneral.MixingModule.mixObjects_cfi import *
mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(25), ## nsec
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   
    input = cms.SecSource("PoolSource",
    nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(19.7)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_3_6_3_SLHC1_patch1/RelValMinBias/GEN-SIM/DESIGN_36_V10_Gauss_29Oct2010_special-v1/0062/F2D78E5C-39E4-DF11-B7F3-003048678B70.root',
#       '/store/relval/CMSSW_3_6_3_SLHC1_patch1/RelValMinBias/GEN-SIM/DESIGN_36_V10_Gauss_29Oct2010_special-v1/0062/9417D37D-4CE4-DF11-926E-003048678F74.root',
#       '/store/relval/CMSSW_3_6_3_SLHC1_patch1/RelValMinBias/GEN-SIM/DESIGN_36_V10_Gauss_29Oct2010_special-v1/0062/5A937DD1-39E4-DF11-A28F-003048678B5E.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0015/F8A2E413-10F3-DF11-B976-0023AEFDF114.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0014/3A8E1661-A5F2-DF11-B932-001A4BD0CF54.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0014/38380477-9FF2-DF11-B9C6-0022199A2E54.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0014/36C92FAA-ACF2-DF11-8C50-00093D114561.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0014/30B07D38-95F2-DF11-A346-001CC4A65D04.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0014/28517D11-9CF2-DF11-8255-001F2908CE36.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/FCD40384-C9F1-DF11-BD86-001EC9DAE365.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/F2C81E6F-42F2-DF11-8E10-001F2908EC96.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/DCECCADE-CDF1-DF11-9509-001EC9F8FCB8.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/D8573868-42F2-DF11-9AAD-001F29087074.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/B43C30BF-C9F1-DF11-9923-0023AEFDEC48.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/B2CBE5E0-D8F1-DF11-9FF5-00093D114763.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/AC5D666F-42F2-DF11-8C30-001F290789D6.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/AA338EC1-CCF1-DF11-8441-001E4F3D88BC.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/9020A679-42F2-DF11-BF34-001F2907F9D2.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/429E9073-42F2-DF11-9D66-001F29082E7E.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/2C48115C-C8F1-DF11-9E6C-0023AEFDE268.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/28D59C2D-C9F1-DF11-A717-0023AEFDE8DC.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/26F20576-42F2-DF11-A265-001F2908AE7C.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/20FFF620-DCF1-DF11-9145-00093D10E0A7.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16_smpx/GEN-SIM/DESIGN_36_V10-v1/0012/1453E3E0-D8F1-DF11-A06D-00093D1148E9.root'
    )
    ),
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        ),
        mixVertices = cms.PSet(
            mixSimVertices
        ),
        mixSH = cms.PSet(
            mixSimHits
        ),
        mixHepMC = cms.PSet(
            mixHepMCProducts
        )
    )
)
