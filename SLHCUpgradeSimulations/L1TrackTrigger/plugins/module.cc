/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, June; 2011, July               ///
///                                      ///
/// Added L1Tracks and ClusterBuilder    ///
/// Unification of Local and Global Stub ///
/// ////////////////////////////////////////

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


/// The Builders

/// The Seed Propagation (Tracking) Algorithms

/// The Hit Matching Algorithms

/// The Clustering Algorithms

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_a.h"
typedef ES_ClusteringAlgorithm_a<Ref_PSimHit_> ClusteringAlgorithm_a_PSimHit_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_a_PSimHit_);
typedef ES_ClusteringAlgorithm_a<Ref_Phase2TrackerDigi_> ClusteringAlgorithm_a_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_a_Phase2TrackerDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_broadside.h"
typedef ES_ClusteringAlgorithm_broadside<Ref_Phase2TrackerDigi_> ClusteringAlgorithm_broadside_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_broadside_Phase2TrackerDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_2d.h"
typedef ES_ClusteringAlgorithm_2d<Ref_Phase2TrackerDigi_> ClusteringAlgorithm_2d_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_2d_Phase2TrackerDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_neighbor.h"
typedef ES_ClusteringAlgorithm_neighbor<Ref_Phase2TrackerDigi_> ClusteringAlgorithm_neighbor_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_neighbor_Phase2TrackerDigi_);

/* - L1 CaloTrigger - */
//#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1CaloTriggerSetupProducer.h"
//DEFINE_FWK_EVENTSETUP_MODULE(L1CaloTriggerSetupProducer);


