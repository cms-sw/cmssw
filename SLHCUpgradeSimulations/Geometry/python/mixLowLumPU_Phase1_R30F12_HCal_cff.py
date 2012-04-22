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
            averageNumber = cms.double(5.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/062F630C-BF5E-E111-BF35-001D0967DE90.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/0A49967B-EB5D-E111-8FF9-0024E87682A6.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/0C14734C-E95D-E111-9171-00A0D1EE9274.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/245D31A0-E45D-E111-AD07-00151796D4B4.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/404215DD-E05D-E111-A07F-0015178C6B54.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/54F05A9D-F55D-E111-842E-0024E87663BA.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/7461FCBC-D05D-E111-BADE-001D0967CFE5.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/7CB36353-D35D-E111-8553-0015178C48FC.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/8670ADDF-EF5D-E111-8B5C-001D0967D319.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/B6939825-EE5D-E111-99F8-0024E876A86F.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/DA753D18-4B5E-E111-965F-00266CF9AB88.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHChcal2-v1/0000/E04689B4-D75D-E111-A3D2-001D0967D56C.root'
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
