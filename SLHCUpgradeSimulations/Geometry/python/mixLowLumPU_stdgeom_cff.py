# sandard geometry minbias pileup files
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
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/FC3F07D7-1395-E011-9D6C-003048679084.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/FAC5DC9C-2B95-E011-B53C-00261894382D.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/F4AD4F5F-1495-E011-981E-003048679228.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/D6B92ED4-1895-E011-8C7F-00304867BECC.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/BE499D41-1A95-E011-BBF6-00261894388D.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/A497BEDC-1095-E011-96A0-002618943875.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/9E5BB94B-1695-E011-87F0-003048678B86.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/82E5DFDD-2B95-E011-B0FE-0030486790A6.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/8294C615-2C95-E011-B8D0-0018F3D0969A.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/8238B044-2995-E011-8C55-0018F3D0969A.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/8093B155-1895-E011-8BCC-003048678FD6.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/66BA7C69-3695-E011-B2CC-00261894391F.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/64A4684C-1695-E011-85D3-003048679228.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/607B2BD1-1495-E011-8B27-003048678FDE.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/5E98E4A0-2F95-E011-AE39-001BFCDBD1BC.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/5CF8A4A2-2D95-E011-85C5-0018F3D096F0.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/5CE92A01-9995-E011-B01F-0018F3D096E6.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/560BF2C7-1795-E011-9A99-003048678BAA.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/46227155-1895-E011-BACE-00304867BECC.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/42E6CA12-3295-E011-A3C2-001A9281173E.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/40DBADB2-2595-E011-85E5-002618943981.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/341A5D29-2C95-E011-A6E9-001A92971AD0.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/2CD6EFCA-1695-E011-A91B-0030486792B4.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/1E7A481E-2A95-E011-AF07-0018F3D0969A.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/163326C2-1A95-E011-BCE4-003048678F74.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/0827FF2C-2D95-E011-9E6A-001A92971B8E.root',
       '/store/relval/CMSSW_4_2_3_patch3/RelValMinBias/GEN-SIM/DESIGN42_V11_110612_special-v1/0092/06BC08D1-1495-E011-99AE-003048679228.root'
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
