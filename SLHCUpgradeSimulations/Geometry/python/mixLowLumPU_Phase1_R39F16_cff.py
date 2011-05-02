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
       '/store/relval/CMSSW_4_2_2_SLHC_pre1/RelValMinBias_14TeV/GEN-SIM/DESIGN42_V10_110429_special-v1/0026/F6A4B42F-6872-E011-906B-0026189438EB.root',
       '/store/relval/CMSSW_4_2_2_SLHC_pre1/RelValMinBias_14TeV/GEN-SIM/DESIGN42_V10_110429_special-v1/0026/E6D8756D-7C72-E011-BC26-001A92971AAA.root',
       '/store/relval/CMSSW_4_2_2_SLHC_pre1/RelValMinBias_14TeV/GEN-SIM/DESIGN42_V10_110429_special-v1/0026/361826B2-6672-E011-9F1E-0026189438AB.root' 
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
