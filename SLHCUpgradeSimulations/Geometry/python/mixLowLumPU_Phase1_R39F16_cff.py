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
       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/EE05606A-A7BD-DF11-AB75-0030486792B6.root',
       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/EA982B5A-A8BD-DF11-8619-0026189438E4.root',
       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/C6EB2D6B-A7BD-DF11-BFB7-00261894387C.root',
       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/72ECC6B1-C5BD-DF11-A15B-0018F3D09658.root',
       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/2EBCB8F0-A9BD-DF11-B817-002618943882.root',
       '/store/relval/CMSSW_3_6_3_SLHC1/RelValMinBias/GEN-SIM-RAW/DESIGN_36_V10_special-v1/0021/04FF2C78-AEBD-DF11-8690-0018F3D0970E.root'
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
