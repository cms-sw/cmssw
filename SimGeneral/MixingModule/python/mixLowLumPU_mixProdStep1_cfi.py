# The following comments couldn't be translated into the new config version:

# E33 cm-2s-1
# mb
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in the low-luminosity phase
from SimGeneral.MixingModule.mixObjects_cfi import *
mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(25), ## nsec
    checktof = cms.bool(False),
    mixProdStep1 = cms.bool(True),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   
    input = cms.SecSource("PoolSource",
	nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(2.8)
        ),
        seed = cms.int32(1234567),
        type = cms.string('poisson'),
        fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_0_pre9/RelValProdMinBias/GEN-SIM-RAW/IDEAL_31X_v1/0007/E067C176-D84E-DE11-BEA6-001617C3B70E.root',
                                          '/store/relval/CMSSW_3_1_0_pre9/RelValProdMinBias/GEN-SIM-RAW/IDEAL_31X_v1/0007/84B81EF5-524F-DE11-9A48-001D09F2543D.root',
                                          '/store/relval/CMSSW_3_1_0_pre9/RelValProdMinBias/GEN-SIM-RAW/IDEAL_31X_v1/0007/167D41FF-D44E-DE11-9F3D-001617C3B6C6.root'
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


