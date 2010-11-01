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
            Lumi = cms.double(10.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_6_3_SLHC1_patch1/RelValMinBias/GEN-SIM/DESIGN_36_V10_Gauss_29Oct2010_special-v1/0062/F2D78E5C-39E4-DF11-B7F3-003048678B70.root',
       '/store/relval/CMSSW_3_6_3_SLHC1_patch1/RelValMinBias/GEN-SIM/DESIGN_36_V10_Gauss_29Oct2010_special-v1/0062/9417D37D-4CE4-DF11-926E-003048678F74.root',
       '/store/relval/CMSSW_3_6_3_SLHC1_patch1/RelValMinBias/GEN-SIM/DESIGN_36_V10_Gauss_29Oct2010_special-v1/0062/5A937DD1-39E4-DF11-A28F-003048678B5E.root'
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
