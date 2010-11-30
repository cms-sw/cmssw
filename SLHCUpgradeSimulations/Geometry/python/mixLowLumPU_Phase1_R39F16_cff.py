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
#       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/EE05606A-A7BD-DF11-AB75-0030486792B6.root',
#       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/EA982B5A-A8BD-DF11-8619-0026189438E4.root',
#       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/C6EB2D6B-A7BD-DF11-BFB7-00261894387C.root',
#       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/72ECC6B1-C5BD-DF11-A15B-0018F3D09658.root',
#       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/2EBCB8F0-A9BD-DF11-B817-002618943882.root',
#       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/04FF2C78-AEBD-DF11-8690-0018F3D0970E.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0014/D623B71A-8DF2-DF11-9D39-0023AEFDEE24.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0014/90AE1115-94F2-DF11-A7B7-001CC4A7C0A4.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0014/7815F2DC-83F2-DF11-8495-80000048FE80.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/F8A78DB0-F3F1-DF11-ADCC-0021286B81DA.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/F6F69F23-F4F1-DF11-AA3C-0021286B8196.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/B26B984C-42F2-DF11-9F9C-001CC4445DC2.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/A81E0686-DEF1-DF11-B597-00093D13C4B1.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/9CC9CE9C-00F2-DF11-B12A-00093D1148DA.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/9C210177-C9F1-DF11-8398-0023AEFDE2A0.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/980038E1-CAF1-DF11-88CD-0023AEFDEC48.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/7A3AC7B0-C3F1-DF11-9CAD-0023AEFDEBD4.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/54DDF94C-C3F1-DF11-B43C-001EC9F8720F.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/4E035AFB-F3F1-DF11-8FDA-0021286B8222.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/469E70A3-3DF2-DF11-A5DF-001CC47B30A0.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/3A7086E1-D8F1-DF11-B17D-00093D113FF6.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/3A6BDED3-3DF2-DF11-9E18-001CC47BEE5E.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/3006A7FB-F3F1-DF11-AA8C-0021286B8212.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/2E7F1595-F2F1-DF11-9C31-0021286B81BA.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/2A575847-42F2-DF11-849D-001CC4445076.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/2A43AA15-C8F1-DF11-A1F2-0023AEFDEE9C.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/241FE043-42F2-DF11-B2BC-001CC4A61CE2.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/1CC1C979-C5F1-DF11-AFB2-001EC9DAED72.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/08978993-F2F1-DF11-A396-0021284F14D2.root',
       '/store/mc/Fall10/MinBias_SLHC_R39F16/GEN-SIM/DESIGN_36_V10-v1/0012/04314B46-42F2-DF11-95DE-001CC47DC95A.root'
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
