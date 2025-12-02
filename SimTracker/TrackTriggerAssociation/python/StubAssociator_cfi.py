# config parameter used by StubAssociator

import FWCore.ParameterSet.Config as cms

StubAssociator_params = cms.PSet (

  InputTagTTStubDetSetVec = cms.InputTag( "TTStubsFromPhase2TrackerDigis", "StubAccepted" ),        # Input TTStubs
  #InputTagTTClusterAssMap = cms.InputTag( "Cleaner", "AtLeastOneCluster" ),                        # TTCluster association map
  InputTagTTClusterAssMap = cms.InputTag( "TTClusterAssociatorFromPixelDigis", "ClusterAccepted" ), # TTCluster association map

  BranchFake = cms.string  ( "UseForFake" ), # name of StubAssociation used for fake rate
  BranchDup  = cms.string  ( "UseForDup"  ), # name of StubAssociation used for duplicate rate
  BranchEff  = cms.string  ( "UseForEff"  ), # name of StubAssociation used for tracking efficiency 

  LooseMatching = cms.bool( False ),
  #LooseMatching = cms.bool( True ),

  MinPt    = cms.double(  2.   ), # pt cut in GeV
  MaxZ0    = cms.double( 15.   ), # half lumi region size in cm
  MaxD0    = cms.double(  1.   ), # cut on impact parameter in cm
  MaxVertR = cms.double(  1.   ), # cut on vertex pos r in cm
  MaxVertZ = cms.double( 30.   ), # cut on vertex pos z in cm

)
