import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in the low-luminosity phase
from SimGeneral.MixingModule.mixObjects_cfi import * 


mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(0),
    minBunch = cms.int32(0), ## in terms of 25 ns

    bunchspace = cms.int32(25), ## nsec
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(True),
    
    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
                   		   
    input = cms.SecSource("PoolSource",	
	type = cms.string('fixed'),
	nbPileupEvents = cms.PSet(
        averageNumber = cms.double(1.0)
    	),
	sequential = cms.untracked.bool(True),
        fileNames =  cms.untracked.vstring('file:PCFLowLumiPU.root')
    ),
    
    mixObjects = cms.PSet(
        # Objects to mix
	mixPCFCH = cms.PSet(
            mixPCFCaloHits
        ),
        mixPCFTracks = cms.PSet(
            mixPCFSimTracks
        ),
        mixPCFVertices = cms.PSet(
            mixPCFSimVertices
        ),
        mixPCFSH = cms.PSet(
            mixPCFSimHits
        ),
        mixPCFHepMC = cms.PSet(
            mixPCFHepMCProducts
        ),
	#add for Step2
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


