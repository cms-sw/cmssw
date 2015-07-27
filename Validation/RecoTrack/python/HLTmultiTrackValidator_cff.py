import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
hltPixelTracksV = hltMultiTrackValidator.clone()
hltPixelTracksV.label                           = cms.VInputTag(cms.InputTag("hltPixelTracks"))
hltPixelTracksV.trackCollectionForDrCalculation = cms.InputTag("hltPixelTracks") 

hltIter0V = hltMultiTrackValidator.clone()
hltIter0V.label                           = cms.VInputTag( cms.InputTag("hltIter0PFlowTrackSelectionHighPurity") )
hltIter0V.trackCollectionForDrCalculation = cms.InputTag("hltIter0PFlowTrackSelectionHighPurity")

hltIter1V = hltMultiTrackValidator.clone()
hltIter1V.label                           = cms.VInputTag( cms.InputTag("hltIter1PFlowTrackSelectionHighPurity") )
hltIter1V.trackCollectionForDrCalculation = cms.InputTag("hltIter1PFlowTrackSelectionHighPurity")

hltIter1MergedV = hltMultiTrackValidator.clone()
hltIter1MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter1Merged") )
hltIter1MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter1Merged")

hltIter2V = hltMultiTrackValidator.clone()
hltIter2V.label                           = cms.VInputTag( cms.InputTag("hltIter2PFlowTrackSelectionHighPurity") )
hltIter2V.trackCollectionForDrCalculation = cms.InputTag("hltIter2PFlowTrackSelectionHighPurity")

hltIter2MergedV = hltMultiTrackValidator.clone()
hltIter2MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter2Merged") )
hltIter2MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter2Merged")

hltIter3V = hltMultiTrackValidator.clone()
hltIter3V.label                           = cms.VInputTag( cms.InputTag("hltIter3PFlowTrackSelectionHighPurity") )
hltIter3V.trackCollectionForDrCalculation = cms.InputTag("hltIter3PFlowTrackSelectionHighPurity")

hltIter3MergedV = hltMultiTrackValidator.clone()
hltIter3MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter3Merged") )
hltIter3MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter3Merged")

hltIter4V = hltMultiTrackValidator.clone()
hltIter4V.label                           = cms.VInputTag( cms.InputTag("hltIter4PFlowTrackSelectionHighPurity") )
hltIter4V.trackCollectionForDrCalculation = cms.InputTag("hltIter4PFlowTrackSelectionHighPurity")

hltIter4MergedV = hltMultiTrackValidator.clone()
hltIter4MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter4Merged") )
hltIter4MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter4Merged")

hltMultiTrackValidation = cms.Sequence(
    hltTPClusterProducer
    + hltTrackAssociatorByHits
    + hltPixelTracksV
    + hltIter0V
    + hltIter1V
    + hltIter1MergedV
    + hltIter2V
    + hltIter2MergedV
#    + hltIter3V
#    + hltIter3MergedV
#    + hltIter4V
#    + hltIter4MergedV
)    
