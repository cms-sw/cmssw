import copy

import FWCore.ParameterSet.Config as cms

from SimTracker.TrackHistory.TrackClassifier_cff import *
from SimTracker.TrackHistory.VertexClassifier_cff import *

def TrackCategorySelector(src, cut):
    trackClassifier.trackProducer = copy.deepcopy(src)
    trackClassifier.trackProducer.setIsTracked(False)
    return cms.EDFilter('TrackSelector', trackClassifier, src = src, cut = cut)

def TrackingParticleCategorySelector(src, cut):
    return cms.EDFilter('TrackingParticleSelector', trackClassifier, src = src, cut = cut)

def VertexCategorySelector(src, cut):
    vertexClassifier.vertexProducer = copy.deepcopy(src)
    vertexClassifier.vertexProducer.setIsTracked(False)
    return cms.EDFilter('VertexSelector', vertexClassifier, src = src, cut = cut)

def TrackingVertexCategorySelector(src, cut):
    return cms.EDFilter('TrackingVertexSelector', vertexClassifier, src = src, cut = cut)

def SecondaryVertexTagInfoCategorySelector(src, pxy, cut):
    vertexClassifier.vertexProducer = copy.deepcopy(pxy)
    vertexClassifier.vertexProducer.setIsTracked(False)    
    return cms.EDFilter('SecondaryVertexTagInfoSelector', vertexClassifier, src = src, cut = cut)
