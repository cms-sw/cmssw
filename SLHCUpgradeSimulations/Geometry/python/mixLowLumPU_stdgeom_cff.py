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
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/E4203C42-9B04-E011-A7CE-002354EF3BCE.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/E0E7B952-9504-E011-AF03-003048678B18.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/D6717442-9304-E011-8E90-002618FDA21D.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/D4A2E054-A904-E011-9FD5-001A928116FE.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/C6B233DE-A204-E011-BC60-001A92810ABA.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/C47592C5-9B04-E011-BD16-002618943959.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/C23A8054-A204-E011-B910-0018F3D096C0.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/BE66C05B-A104-E011-A59F-001BFCDBD1BC.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/B840F758-A304-E011-9A3F-001A92810ABA.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/B66F0FC2-9A04-E011-BFED-0026189438D3.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/B4D5BCC7-9204-E011-8515-0026189438F9.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/B22789C4-9704-E011-A4FE-003048D15E02.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/AE310A63-9F04-E011-89B7-0018F3D09680.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/ACC83145-A104-E011-8AB7-001A92810A98.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/96311464-9304-E011-875F-0026189438BD.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/8849274B-9404-E011-A04C-0026189438CE.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/8478D057-9A04-E011-8EE1-001A92810AAE.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/781653E7-9604-E011-897D-003048D15E02.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/66DFE744-9804-E011-8407-00304867902E.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/6618FD02-A504-E011-81B0-0018F3D096D2.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/64D0AB4C-9704-E011-B4BA-003048D15E02.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/642DBACD-AF04-E011-B839-002618FDA259.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/5E5C69C4-9504-E011-B85C-00304867D446.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/485A5ACB-9C04-E011-B80E-002618FDA265.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/40BD6FCD-9804-E011-BAF2-00304867902E.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/2EDCA4C4-A004-E011-AB0B-001A92810A98.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/269BE108-A504-E011-B0A8-001A92811726.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/1806E3E0-9604-E011-8B03-003048678E94.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/0CB095CE-9404-E011-91B5-0026189438DA.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/04305755-A704-E011-95BF-001A9281174A.root',
       '/store/relval/CMSSW_3_6_3_patch2/RelValMinBias/GEN-SIM/DESIGN_36_V10_special-v1/0106/00EB1558-9304-E011-AEDF-002618FDA208.root'
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
