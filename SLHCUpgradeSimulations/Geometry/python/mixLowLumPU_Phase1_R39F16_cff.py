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
	'/store/relval/CMSSW_4_2_3_SLHC_pre1/RelValMinBias/GEN-SIM/DESIGN42_V11_20110605_special-v1/0026/8E6796DD-E37F-E011-8E75-0018F3D09604.root',
	'/store/relval/CMSSW_4_2_3_SLHC_pre1/RelValMinBias/GEN-SIM/DESIGN42_V11_20110605_special-v1/0026/8495EEF0-E07F-E011-BBCE-001BFCDBD11E.root',
	'/store/relval/CMSSW_4_2_3_SLHC_pre1/RelValMinBias/GEN-SIM/DESIGN42_V11_20110605_special-v1/0026/12510EDE-8980-E011-8A91-0018F3D0960E.root' 
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
