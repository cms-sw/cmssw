# The following comments couldn't be translated into the new config version:

# E33 cm-2s-1
# mb
import FWCore.ParameterSet.Config as cms

# this is the configuration to model pileup in StageA with 156x156 bunchcrossings
from SimGeneral.MixingModule.mixObjects_cfi import *
mix = cms.EDProducer("MixingModule",
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 nsec

    bunchspace = cms.int32(450), ## ns
	
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),
    	
    playback = cms.untracked.bool(False),
    input = cms.SecSource("EmbeddedRootSource",
        nbPileupEvents = cms.PSet(
            sigmaInel = cms.double(80.0),
            Lumi = cms.double(0.1615)
        ),
        type = cms.string('poisson'),
	sequential = cms.untracked.bool(False),
        fileNames = cms.untracked.vstring('/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/00233C31-5806-DD11-9DDC-001617DBD5B2.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/3469E801-5C06-DD11-93DC-00304885AE42.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/52B1C4F8-5406-DD11-8031-00304885B130.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/6012E6A6-6106-DD11-B9C6-003048562890.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/66D7FE91-5606-DD11-A3C7-00304885A74E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/B6CDB304-5706-DD11-B9E4-001617C3B76E.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/DC900058-5506-DD11-96DD-000423D98AF0.root', 
            '/store/relval/2008/4/9/RelVal-RelValMinBias-1207754630/0002/F80E6D8B-5706-DD11-B8A9-001617C3B76E.root')
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


