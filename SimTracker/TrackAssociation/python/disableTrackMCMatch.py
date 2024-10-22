import FWCore.ParameterSet.Config as cms
#
def disableTrackMCMatch(process):
    
    if hasattr(process,'prunedTpClusterProducer'):
        process.prunedTpClusterProducer.throwOnMissingCollections = cms.bool(False)
        
    if hasattr(process,'prunedTrackMCMatch'):
        process.prunedTrackMCMatch.throwOnMissingTPCollection = cms.bool(False)
        
    return process
