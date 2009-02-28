import FWCore.ParameterSet.Config as cms

from SimTracker.TrackHistory.TrackClassifier_cff import *
from SimTracker.TrackHistory.VertexClassifier_cff import *

def TrackCategorySelector(select, source=''):

    if source == '':
        return cms.EDFilter('TrackSelector',
            trackClassifier,
            src = cms.InputTag(trackClassifier.trackProducer), 
            select = cms.untracked.string(select)
        )

    trackClassifier.trackProducer = cms.untracked.InputTag(source)
    return cms.EDFilter('TrackSelector', 
        trackClassifier, 
        src = cms.InputTag(source), 
        select = cms.untracked.string(select)
    )

def VertexCategorySelector(select, source=''):

    if source == '':
        return cms.EDFilter('VertexSelector',
            vertexClassifier,
            src = cms.InputTag(VertexClassifier.vertexProducer),
            select = cms.untracked.string(select)
        )

    vertexClassifier.vertexProducer = cms.untracked.InputTag(source)
    return cms.EDFilter('VertexSelector',
        vertexClassifier,
        src = cms.InputTag(source),
        select = cms.untracked.string(select)
    )

