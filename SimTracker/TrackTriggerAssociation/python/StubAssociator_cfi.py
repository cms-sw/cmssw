import FWCore.ParameterSet.Config as cms

StubAssociator_params = cms.PSet (
  InputTagTTStubDetSetVec = cms.InputTag( "TTStubsFromPhase2TrackerDigis",     "StubAccepted"      ), #
  InputTagTTClusterAssMap = cms.InputTag( "TTClusterAssociatorFromPixelDigis", "ClusterAccepted"   ), #
  #InputTagTTClusterAssMap = cms.InputTag( "CleanAssoc", "AtLeastOneCluster"   ),
  BranchReconstructable   = cms.string  ( "Reconstructable" ),                                        # name of StubAssociation collection made with reconstractable TPs
  BranchSelection         = cms.string  ( "UseForAlgEff"    ),                                        # name of StubAssociation collection used for tracking efficiency 

  MinPt           = cms.double(  2.   ), # pt cut in GeV
  MaxEta0         = cms.double(  2.4  ), # max eta for TP with z0 = 0
  MaxZ0           = cms.double( 15.   ), # half lumi region size in cm
  MaxD0           = cms.double(  5.   ), # cut on impact parameter in cm
  MaxVertR        = cms.double(  1.   ), # cut on vertex pos r in cm
  MaxVertZ        = cms.double( 30.   ), # cut on vertex pos z in cm
  MinLayers       = cms.int32 (  4    ), # required number of associated stub layers to a TP to consider it reconstruct-able
  MinLayersPS     = cms.int32 (  0    ), # required number of associated ps stub layers to a TP to consider it reconstruct-able
  MinLayersGood   = cms.int32 (  4    ), # required number of layers a found track has to have in common with a TP to consider it matched
  MinLayersGoodPS = cms.int32 (  0    ), # required number of ps layers a found track has to have in common with a TP to consider it matched
  MaxLayersBad    = cms.int32 (  1    ), # max number of unassociated 2S stubs allowed to still associate TTTrack with TP
  MaxLayersBadPS  = cms.int32 (  0    )  # max number of unassociated PS stubs allowed to still associate TTTrack with TP

)
