import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.MassReplace import massReplaceInputTag as MassReplaceInputTag
#
# mass replace OfflinePrimaryVertices with OfflinePrimaryVerticesWithBS 
# (doesn't affect defaults in the source code, e.g. provided by fillDescriptions)
#
def massReplaceOfflinePrimaryVerticesToUseBeamSpot(process):
    # swap all occurrences
    process = MassReplaceInputTag(process,"offlinePrimaryVertices","offlinePrimaryVerticesWithBS")
    
    # excepted of course for the primary source...
    if hasattr(process,'offlinePrimaryVerticesWithBS'):
        process.offlinePrimaryVerticesWithBS.src = cms.InputTag("offlinePrimaryVertices","WithBS")
        
    return process

#
# makes OfflinePrimaryVertices equivalent to OfflinePrimaryVerticesWithBS
# by changing the input vertices collection of the sorted PV
# see file https://github.com/cms-sw/cmssw/blob/master/RecoVertex/Configuration/python/RecoVertex_cff.py
# 
def swapOfflinePrimaryVerticesToUseBeamSpot(process):
    if hasattr(process,'offlinePrimaryVertices'):
        process.offlinePrimaryVertices.vertices="unsortedOfflinePrimaryVertices:WithBS"

    return process
