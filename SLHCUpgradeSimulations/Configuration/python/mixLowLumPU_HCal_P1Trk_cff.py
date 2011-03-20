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
            averageNumber = cms.double(25.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/F841098A-8651-E011-8E4D-001A9281172C.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/EE6EFFFA-8D51-E011-8F85-0018F3D09654.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/C69759EC-8251-E011-B5D1-0018F3D096C8.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/B2B0B5F8-6F51-E011-BE61-00261894387E.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/B0782A5C-8B51-E011-A982-001731EF61B4.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/B0120110-8551-E011-A3BE-001A9281172C.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/AE92A98E-6751-E011-BA27-002618943964.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/A25627B5-8851-E011-959B-0018F3D09658.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/A0DA62BF-8451-E011-9605-001A9281172C.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/9ECFA040-6451-E011-9A23-002618943862.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/98402E14-6651-E011-82BE-002618943884.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/90972C38-8651-E011-A880-0018F3D0968C.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/8C51F021-8251-E011-B391-001A92971B06.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/84803A96-8351-E011-9D29-001A92810AAE.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/84706DBA-6451-E011-91E9-00261894386F.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/8211833D-8751-E011-96B1-0018F3D0962A.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/66E39E9F-6C51-E011-A9B7-00261894388D.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/6603803D-6951-E011-99D3-00304867D836.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/62E86CD6-6651-E011-9676-003048D3C010.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/48A83ED9-8751-E011-B350-0018F3D096E6.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/3E210869-8351-E011-BCFF-0018F3D09624.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/2E6463D2-8151-E011-92AA-001A92810AAE.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/1E8F13EC-6A51-E011-B279-00304867924E.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/1E69639A-6551-E011-ADCE-00261894387E.root',
       '/store/relval/CMSSW_3_6_3_SLHC3_patch1/RelValMinBias_14TeV/GEN-SIM/DESIGN_36_V10_Gauss_special-v1/0013/0A86A74A-8151-E011-932B-001A92971B06.root' 
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
