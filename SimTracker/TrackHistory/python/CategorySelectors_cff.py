import copy

import FWCore.ParameterSet.Config as cms

from SimTracker.TrackHistory.MuonClassifier_cff import *
from SimTracker.TrackHistory.TrackClassifier_cff import *
from SimTracker.TrackHistory.VertexClassifier_cff import *

def MuonCategorySelector(src, cut):
    trackClassifier.trackProducer = copy.deepcopy(src)
    trackClassifier.trackProducer.setIsTracked(False)
    return cms.EDFilter('TrackCategorySelector', MuonClassifier, src = src, cut = cut)

def TrackCategorySelector(src, cut):
    trackClassifier.trackProducer = copy.deepcopy(src)
    trackClassifier.trackProducer.setIsTracked(False)
    return cms.EDFilter('TrackCategorySelector', trackClassifier, src = src, cut = cut)

def TrackingParticleCategorySelector(src, cut):
    trackClassifier.enableRecoToSim = cms.untracked.bool(False);
    trackClassifier.enableSimToReco = cms.untracked.bool(False);    
    return cms.EDFilter('TrackingParticleCategorySelector', trackClassifier, src = src, cut = cut)

def VertexCategorySelector(src, cut):
    vertexClassifier.vertexProducer = copy.deepcopy(src)
    vertexClassifier.vertexProducer.setIsTracked(False)
    return cms.EDFilter('VertexCategorySelector', vertexClassifier, src = src, cut = cut)

def TrackingVertexCategorySelector(src, cut):
    vertexClassifier.enableRecoToSim = cms.untracked.bool(False);
    vertexClassifier.enableSimToReco = cms.untracked.bool(False);    
    return cms.EDFilter('TrackingVertexCategorySelector', vertexClassifier, src = src, cut = cut)

def SecondaryVertexTagInfoCategorySelector(src, pxy, cut):
    vertexClassifier.vertexProducer = copy.deepcopy(pxy)
    vertexClassifier.vertexProducer.setIsTracked(False)    
    return cms.EDFilter('SecondaryVertexTagInfoCategorySelector', vertexClassifier, src = src, cut = cut)
# foo bar baz
# KgDkT5x8CrPy6
# I5ocz1hC4siy9
