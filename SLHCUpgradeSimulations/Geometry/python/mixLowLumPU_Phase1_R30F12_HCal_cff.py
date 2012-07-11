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
            averageNumber = cms.double(50.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
# Useing Block: /MinBias_TuneZ2star_14TeV-pythia6/Summer12-UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/GEN-SIM#e53ae8c2-a36c-11e1-903a-00221959e72f 233,280 events
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/EAF214C2-C9A3-E111-B324-18A90570BB1C.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/1ACAF486-5FA3-E111-BA2A-18A905705B70.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/5A8DFB50-95A3-E111-B471-18A905705A8C.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/70C46D55-DBA3-E111-9626-18A905705B70.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/7AE0360B-C8A3-E111-BB3C-18A90570BB1C.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/8A32AD02-99A3-E111-BF84-18A90570394C.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/20755E6A-B1A3-E111-B3DF-18A90570BB52.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/9466F494-AEA3-E111-B558-18A90570BB1C.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/B4DE1D91-5FA3-E111-A792-18A905702BD2.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/CCCC71A6-95A3-E111-9E0C-18A90570A7EC.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/D006DF55-95A3-E111-B84A-18A90570497E.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/307E9292-5FA3-E111-B42E-18A905702BD2.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/6873D550-95A3-E111-8BF9-18A905705B70.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/82E3035F-97A3-E111-83D2-18A905705B70.root',
'/store/mc/Summer12/MinBias_TuneZ2star_14TeV-pythia6/GEN-SIM/UpgradeHCAL_PixelPhase1-DESIGN42_V17-v2/0000/92B94B51-95A3-E111-94CB-18A905706BA8.root'
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
