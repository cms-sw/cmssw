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
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/FA86F8B0-6592-E011-AA1E-002618943834.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/F883F3E8-5F92-E011-B8ED-0026189438E9.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/F213890C-6392-E011-8701-002618943983.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/E8B771B7-6292-E011-9EB3-00261894395C.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/CCC19FC5-6092-E011-AA11-002618943925.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/CA29DD2C-6092-E011-95E8-002618943882.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/C8F50E0C-6292-E011-90C9-002618943947.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/C29D4635-5F92-E011-A7BA-002618943919.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/BCA793F4-6292-E011-9C35-0026189438C4.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/BAB5E802-6292-E011-8643-002618943951.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/B4E335C0-B192-E011-BA7C-00261894382A.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/AE2E147A-6192-E011-A796-002618943954.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/AA2621A6-6192-E011-A1F2-002618943985.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/A6FE4EBC-6292-E011-B212-002618943916.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/A4236922-6092-E011-8968-002618943951.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/965D6ADF-6192-E011-8FFE-002618943970.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/8625B32B-6092-E011-81A1-002618943947.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/84CCAD21-6192-E011-B652-002618943954.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/80EDD728-5E92-E011-8C52-00261894390E.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/7A9AA59E-5F92-E011-B461-00261894393B.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/764E0B16-6492-E011-978C-00261894394B.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/6E100E39-6192-E011-98B9-002618943925.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/625D68B5-6292-E011-9125-0026189438C4.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/5A6ED12A-6292-E011-9E6D-0026189438DA.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/4CAFDC20-6092-E011-9899-0026189438F8.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/4223C10E-6092-E011-9BF2-002618943919.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/40DEFE2C-6492-E011-8863-0026189437EB.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/3605B733-6092-E011-BEB3-00261894396D.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/2637D045-5E92-E011-82AF-002618943877.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/16909893-6392-E011-8D57-002618943946.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/06CDAD21-6192-E011-8C6F-002618943954.root',
       '/store/relval/CMSSW_4_2_3_SLHC3/RelValMinBias/GEN-SIM/DESIGN42_V11_110608_special-v1/0092/02263F95-6192-E011-A798-002618943925.root'
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
