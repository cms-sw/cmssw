import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cfi import *
hltPixelTracksV = hltMultiTrackValidator.clone()
hltPixelTracksV.label                           = cms.VInputTag(cms.InputTag("hltPixelTracks"))
hltPixelTracksV.associatormap                   = cms.InputTag ( "tpToHLTpixelTrackAssociation" ) 
hltPixelTracksV.trackCollectionForDrCalculation = cms.InputTag("hltPixelTracks") 

hltIter0V = hltMultiTrackValidator.clone()
hltIter0V.label                           = cms.VInputTag( cms.InputTag("hltIter0PFlowTrackSelectionHighPurity") )
hltIter0V.associatormap                   = cms.InputTag( "tpToHLTiter0HPtracksAssociation" )
hltIter0V.trackCollectionForDrCalculation = cms.InputTag("hltIter0PFlowTrackSelectionHighPurity")

hltIter1V = hltMultiTrackValidator.clone()
hltIter1V.label                           = cms.VInputTag( cms.InputTag("hltIter1PFlowTrackSelectionHighPurity") )
hltIter1V.associatormap                   = cms.InputTag ( "tpToHLTiter1HPtracksAssociation" ) 
hltIter1V.trackCollectionForDrCalculation = cms.InputTag("hltIter1PFlowTrackSelectionHighPurity")

hltIter1MergedV = hltMultiTrackValidator.clone()
hltIter1MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter1Merged") )
hltIter1MergedV.associatormap                   = cms.InputTag ( "tpToHLTiter1MergedTracksAssociation" ) 
hltIter1MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter1Merged")

hltIter2V = hltMultiTrackValidator.clone()
hltIter2V.label                           = cms.VInputTag( cms.InputTag("hltIter2PFlowTrackSelectionHighPurity") )
hltIter2V.associatormap                   = cms.InputTag ( "tpToHLTiter2HPtracksAssociation" ) 
hltIter2V.trackCollectionForDrCalculation = cms.InputTag("hltIter2PFlowTrackSelectionHighPurity")

hltIter2MergedV = hltMultiTrackValidator.clone()
hltIter2MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter2Merged") )
hltIter2MergedV.associatormap                   = cms.InputTag ("tpToHLTiter2MergedTracksAssociation" )  
hltIter2MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter2Merged")

hltIter3V = hltMultiTrackValidator.clone()
hltIter3V.label                           = cms.VInputTag( cms.InputTag("hltIter3PFlowTrackSelectionHighPurity") )
hltIter3V.associatormap                   = cms.InputTag( "tpToHLTiter3HPtracksAssociation" )
hltIter3V.trackCollectionForDrCalculation = cms.InputTag("hltIter3PFlowTrackSelectionHighPurity")

hltIter3MergedV = hltMultiTrackValidator.clone()
hltIter3MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter3Merged") )
hltIter3MergedV.associatormap                   = cms.InputTag ( "tpToHLTiter3MergedTracksAssociation" ) 
hltIter3MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter3Merged")

hltIter4V = hltMultiTrackValidator.clone()
hltIter4V.label                           = cms.VInputTag( cms.InputTag("hltIter4PFlowTrackSelectionHighPurity") )
hltIter4V.associatormap                   = cms.InputTag( "tpToHLTiter4HPtracksAssociation" )
hltIter4V.trackCollectionForDrCalculation = cms.InputTag("hltIter4PFlowTrackSelectionHighPurity")

hltIter4MergedV = hltMultiTrackValidator.clone()
hltIter4MergedV.label                           = cms.VInputTag( cms.InputTag("hltIter4Merged") )
hltIter4MergedV.associatormap                   = cms.InputTag ( "tpToHLTiter4MergedTracksAssociation" ) 
hltIter4MergedV.trackCollectionForDrCalculation = cms.InputTag("hltIter4Merged")

from Validation.RecoTrack.cutsTPEffic_cfi import *
from Validation.RecoTrack.cutsTPFake_cfi import *

from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *

hltMultiTrackValidation = cms.Sequence(
#    simHitTPAssocProducer
#    +
    hltTPClusterProducer
    + tpToHLTtracksAssociationSequence
    + cms.ignore(cutsTPEffic)
    + cms.ignore(cutsTPFake)
    + hltPixelTracksV
    + hltIter0V
    + hltIter1V
    + hltIter1MergedV
    + hltIter2V
    + hltIter2MergedV
    + hltIter3V
    + hltIter3MergedV
    + hltIter4V
    + hltIter4MergedV
)    

