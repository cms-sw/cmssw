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
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/F0B4FE53-D38E-E011-B36E-0030487CD17C.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/EC4D4774-D08E-E011-9061-0030487CD17C.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/E842A762-DC8E-E011-B202-0030487CD6DA.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/CE8F4E12-DF8E-E011-BE92-0030487CD6D2.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/CC2DD893-D88E-E011-989B-001D09F24024.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/C43FDAA9-DE8E-E011-B75F-0030487A17B8.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/C0E41953-E18E-E011-80A0-0030487CD812.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/C02D7E76-DE8E-E011-9C92-0030487CD17C.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/BC8CB6CD-DF8E-E011-A6F1-0030487A17B8.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/BA54B495-D18E-E011-A0A7-0030487CD17C.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/B6FFAD65-E38E-E011-8B5B-001D09F24FEC.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/B2D72F93-DD8E-E011-B444-0030487CD6DA.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/AAF8E0A9-DE8E-E011-8F42-0030487CD6DA.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/A48B32AF-E18E-E011-B542-0030487CBD0A.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/922EE7CE-E08E-E011-8226-0030487C608C.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/8AAF30ED-D28E-E011-8596-00304879BAB2.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/82FE04BD-698F-E011-BE5E-001D09F2A690.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/82143D58-D68E-E011-A486-0030487CD718.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/7C746CBE-D28E-E011-8F3D-00304879BAB2.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/6A3CC9BC-DF8E-E011-91A7-0030487CD6E8.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/624A965B-D28E-E011-A3AA-0030487CD6DA.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/5E349033-E38E-E011-A575-0030487C7E18.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/549609F0-DC8E-E011-BF9F-0030487CD6DA.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/467EA83E-D48E-E011-9BEE-001D09F28D4A.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/28FB5932-D58E-E011-B451-0030487CAEAC.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/16108543-DC8E-E011-8196-0030487CD812.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/1411505C-E08E-E011-AD4F-0030487CD6E8.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/0C8D223A-DF8E-E011-AE4A-0030487A17B8.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/0A814D19-DE8E-E011-805A-0030487CD6DA.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/068D36D3-E28E-E011-8528-0030487CD6D2.root',
       '/store/relval/CMSSW_4_2_3_SLHC2/RelValMinBias/GEN-SIM/DESIGN42_V11_110603_special-v1/0000/04988117-D78E-E011-9792-0030487CD6DA.root' 
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
