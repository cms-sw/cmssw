import copy

import FWCore.ParameterSet.Config as cms

from SimTracker.TrackHistory.TrackClassifier_cff import *
from SimTracker.TrackHistory.VertexClassifier_cff import *

def TrackCategorySelector(cut, src):
    trackClassifier.trackProducer = copy.deepcopy(src)
    trackClassifier.trackProducer.setIsTracked(False)
    return cms.EDFilter('TrackSelector', trackClassifier, src = src, cut = cut)

def VertexCategorySelector(cut, src):
    vertexClassifier.vertexProducer = copy.deepcopy(src)
    vertexClassifier.vertexProducer.setIsTracked(False)
    return cms.EDFilter('VertexSelector', vertexClassifier, src = src, cut = cut)

