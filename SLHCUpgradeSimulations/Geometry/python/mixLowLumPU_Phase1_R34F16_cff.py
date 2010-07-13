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
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(10.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('/store/user/cheung/phase1/r34v25/minbias4pu/MergedMinBias_GEN_SIM_01.root',
        '/store/user/cheung/phase1/r34v25/minbias4pu/MergedMinBias_GEN_SIM_02.root'
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
