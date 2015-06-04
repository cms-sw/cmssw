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
            averageNumber = cms.double(5.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/1423B147-CE32-E111-AF5F-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/30801B50-D332-E111-9062-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/38C6C5AD-9632-E111-BD75-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/3E75FBA1-0733-E111-AD72-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/542B27AD-9632-E111-8B83-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/62F210AD-9632-E111-9E98-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/781BE5F6-CF32-E111-BDE1-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/784B6556-9832-E111-B23D-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/7AA5769C-0733-E111-9666-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/82A32BB7-9B32-E111-97D5-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/981C00AE-9632-E111-BF4B-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/9AC5369C-0733-E111-BC1F-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/9AED0FAD-9632-E111-990A-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/A049CB9F-D132-E111-A44E-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/C076FCAD-D632-E111-8BBF-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/D0FD5EA1-0733-E111-BC8D-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/D0FDA0E7-CA32-E111-8232-0002C90A3678.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/D6D405B7-9B32-E111-B901-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/D825ACA6-0733-E111-9D27-0002C90B7F8E.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/E0A6EE54-9832-E111-8A8E-0002C90A3426.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/E0C83145-9332-E111-B239-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/E8C0CDA2-0733-E111-91E5-0002C90B743A.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/F07A7AE8-8F32-E111-B342-0002C90A3690.root',
       '/store/mc/Summer11/MinBias/GEN-SIM/DESIGN42_V11_428_SLHChcal-v1/0003/F0CCC254-9832-E111-B4DE-0002C90A3426.root' 
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
