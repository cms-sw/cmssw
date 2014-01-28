import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.TrackTrigger_cff import *
from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *

L1TrackTrigger=cms.Sequence(TrackTriggerClustersStubs*TrackTriggerAssociatorClustersStubs*TrackTriggerTTTracks*TrackTriggerAssociatorTracks)


#from Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi import *
#from SLHCUpgradeSimulations.L1TrackTrigger.ClusteringAlgorithmRegister_cfi import *
#from SLHCUpgradeSimulations.L1TrackTrigger.HitMatchingAlgorithmRegister_cfi import *
#from SLHCUpgradeSimulations.L1TrackTrigger.Cluster_cfi import *
#from SLHCUpgradeSimulations.L1TrackTrigger.Stub_cfi import *
#
#ClusteringAlgorithm_PixelDigi_ = cms.ESPrefer('ClusteringAlgorithm_broadside_PixelDigi_')
#HitMatchingAlgorithm_PixelDigi_ = cms.ESPrefer('HitMatchingAlgorithm_window2013_PixelDigi_')
#
##move these to the cfis
#L1TkClustersFromPixelDigis.rawHits = cms.VInputTag(cms.InputTag("simSiPixelDigis"))
#L1TkStubsFromPixelDigis.L1TkClusters = cms.InputTag("L1TkClustersFromPixelDigis")
#HitMatchingAlgorithm_window2012_PixelDigi_.minPtThreshold = cms.double(2.0)
#
#from SLHCUpgradeSimulations.L1TrackTrigger.TrackingAlgorithmRegister_cfi import *
#from SLHCUpgradeSimulations.L1TrackTrigger.Track_cfi import *
#TrackingAlgorithm_PixelDigi_ = cms.ESPrefer('TrackingAlgorithm_exactBarrelEndcap_PixelDigi_')
##TrackingAlgorithm_PixelDigi_ = cms.ESPrefer('TrackingAlgorithm_exactLongBarrel_PixelDigi_')

##and the sequence to run
#L1TrackTrigger = cms.Sequence(L1TkClustersFromPixelDigis*L1TkStubsFromPixelDigis)



