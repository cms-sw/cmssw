# Phase 1 R34V25 minbias pileup files
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
                   
    input = cms.SecSource("EmbeddedRootSource",
    nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(10.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/26563C5F-7D32-E111-BB34-0002C90B7F8E.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/2EBE1910-7F32-E111-92E1-0002C90A3460.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/30AE4100-7A32-E111-A754-0002C90B7F8E.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/3412CADC-C532-E111-8FAA-0002C90B3976.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/3849055F-7D32-E111-9ACB-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/62BB2910-7F32-E111-B15E-0002C90A3460.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/6A2F2BE1-8A32-E111-B881-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/7659200D-7A32-E111-B49C-0002C90B3990.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/84DE9D53-7832-E111-B152-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/9CD6AF54-7832-E111-94D9-0002C90B7F8E.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/B43BD901-7A32-E111-9815-0002C90B3990.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/C0DBFE5E-7D32-E111-B685-0002C90B7F8E.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/CADFD60D-7A32-E111-9C49-0002C90B3968.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/DCAC3809-7A32-E111-89E0-0002C90B7F8E.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/E23C306E-8232-E111-8B73-0002C90B7488.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/F8003E54-7832-E111-8A5F-0002C90B7F8E.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/FA46760E-7A32-E111-841C-0002C90B3968.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHC1-v1/0003/FED72A53-7832-E111-9F34-0002C90B743A.root'
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
