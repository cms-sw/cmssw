import FWCore.ParameterSet.Config as cms

StubAssociator_params = cms.PSet (
  InputTagTTStubDetSetVec = cms.InputTag( "TTStubsFromPhase2TrackerDigis",     "StubAccepted"    ), #
  InputTagTTClusterAssMap = cms.InputTag( "TTClusterAssociatorFromPixelDigis", "ClusterAccepted" ), #
  InputTagTTStubAssMap    = cms.InputTag( "TTStubAssociatorFromPixelDigis",    "StubAccepted"    ), #
  BranchReconstructable   = cms.string  ( "Reconstructable" ),                                      # name of StubAssociation collection made with reconstractable TPs
  BranchSelection         = cms.string  ( "UseForAlgEff"    )                                       # name of StubAssociation collection used for tracking efficiency
)