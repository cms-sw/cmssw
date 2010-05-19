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
            Lumi = cms.double(10.0)
        ),
        type = cms.string('poisson'),
    sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-01.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-02.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-03.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-04.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-05.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-06.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-07.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-08.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-09.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-10.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-11.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-12.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-13.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-14.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-15.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-16.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-17.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-18.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-19.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-20.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-21.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-22.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-23.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-24.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-25.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-26.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-27.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-28.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-29.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-30.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-31.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-32.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-33.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-34.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-35.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-36.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-37.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-38.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-39.root',
        'file:/uscms_data/d2/cheung/slhc/stdgeom/minbias4pu/PythiaMinBias_GEN_SIM-40.root'
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
