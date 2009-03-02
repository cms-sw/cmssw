import FWCore.ParameterSet.Config as cms

from SimTracker.TrackHistory.TrackClassifier_cff import *
from SimTracker.TrackHistory.VertexClassifier_cff import *

def TrackCategorySelector(cut, src = ''):
 
    if src == '':
        return cms.EDFilter('TrackSelector',
            trackClassifier,
            src = cms.InputTag(trackClassifier.trackProducer), 
            cut = cms.string(cut)
        )

    trackClassifier.trackProducer = cms.untracked.InputTag(src)
    return cms.EDFilter('TrackSelector', 
        trackClassifier, 
        src = cms.InputTag(src), 
        cut = cms.untracked.string(cut)
    )

def VertexCategorySelector(cut, src = ''):

    if src == '':
        return cms.EDFilter('VertexSelector',
            vertexClassifier,
            src = cms.InputTag(VertexClassifier.vertexProducer),
            cut = cms.string(cut)
        )

    vertexClassifier.vertexProducer = cms.untracked.InputTag(src)
    return cms.EDFilter('VertexSelector',
        vertexClassifier,
        src = cms.InputTag(src),
        cut = cms.string(cut)
    )

