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
                   
    input = cms.SecSource("PoolSource",
    nbPileupEvents = cms.PSet(
	    averageNumber = cms.double(50.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/02FA8B68-EC5D-E111-8D2B-001D0967D6AC.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/3E284576-925E-E111-B4F4-00266CF9B1C4.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/42ACBB89-2B5E-E111-86DB-0015178C4D1C.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/4AB2F686-E55D-E111-B514-00A0D1EEAAA0.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/5A1E2A50-EA5D-E111-A642-001D0967DF2B.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/7A01B3E8-ED5D-E111-A22F-001D0967D535.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/84BA90FC-F75D-E111-AC46-00A0D1EEC298.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/969782E5-F15D-E111-9586-0024E876A83B.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/AAE82357-D45D-E111-A5C9-00151796C12C.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/C09D6B1D-E75D-E111-A0D2-00A0D1EE8E60.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/CCC10265-E25D-E111-B689-001D0967D341.root',
       '/store/mc/Summer12/MinBias_14TeV/GEN-SIM/DESIGN42_V17_SLHCTk-v1/0000/FAF5B2D7-F35D-E111-9980-0024E8767D11.root'
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
