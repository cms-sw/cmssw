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

/// The Builders

//#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkBeamBuilder.h"
//DEFINE_FWK_MODULE(L1TkBeamBuilder);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkClusterBuilder.h"
typedef L1TkClusterBuilder<Ref_PSimHit_> L1TkClusterBuilder_PSimHit_;
DEFINE_FWK_MODULE(L1TkClusterBuilder_PSimHit_);
typedef L1TkClusterBuilder<Ref_PixelDigi_> L1TkClusterBuilder_PixelDigi_;
DEFINE_FWK_MODULE(L1TkClusterBuilder_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkStubBuilder.h"
typedef L1TkStubBuilder<Ref_PSimHit_> L1TkStubBuilder_PSimHit_;
DEFINE_FWK_MODULE(L1TkStubBuilder_PSimHit_);
typedef L1TkStubBuilder<Ref_PixelDigi_> L1TkStubBuilder_PixelDigi_;
DEFINE_FWK_MODULE(L1TkStubBuilder_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1TkTrackBuilder.h"
typedef L1TkTrackBuilder<Ref_PixelDigi_> L1TkTrackBuilder_PixelDigi_;
DEFINE_FWK_MODULE(L1TkTrackBuilder_PixelDigi_);

/// The Seed Propagation (Tracking) Algorithms
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm_exactLongBarrel.h"
typedef ES_TrackingAlgorithm_exactLongBarrel<Ref_PixelDigi_> TrackingAlgorithm_exactLongBarrel_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(TrackingAlgorithm_exactLongBarrel_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm_exactBarrelEndcap.h"
typedef ES_TrackingAlgorithm_exactBarrelEndcap<Ref_PixelDigi_> TrackingAlgorithm_exactBarrelEndcap_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(TrackingAlgorithm_exactBarrelEndcap_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/TrackingAlgorithm_associativeBarrelEndcap.h"
typedef ES_TrackingAlgorithm_associativeBarrelEndcap<Ref_PixelDigi_> TrackingAlgorithm_associativeBarrelEndcap_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(TrackingAlgorithm_associativeBarrelEndcap_PixelDigi_);


/// The Hit Matching Algorithms

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_a.h"
typedef ES_HitMatchingAlgorithm_a<Ref_PSimHit_> HitMatchingAlgorithm_a_PSimHit_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_a_PSimHit_);
typedef ES_HitMatchingAlgorithm_a<Ref_PixelDigi_> HitMatchingAlgorithm_a_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_a_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_thresholds.h"
typedef ES_HitMatchingAlgorithm_thresholds<Ref_PixelDigi_> HitMatchingAlgorithm_thresholds_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_thresholds_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_window.h"
typedef ES_HitMatchingAlgorithm_window<Ref_PixelDigi_> HitMatchingAlgorithm_window_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_window_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_globalgeometry.h"
typedef ES_HitMatchingAlgorithm_globalgeometry<Ref_PSimHit_> HitMatchingAlgorithm_globalgeometry_PSimHit_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_globalgeometry_PSimHit_);
typedef ES_HitMatchingAlgorithm_globalgeometry<Ref_PixelDigi_> HitMatchingAlgorithm_globalgeometry_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_globalgeometry_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_pixelray.h"
typedef ES_HitMatchingAlgorithm_pixelray<Ref_PixelDigi_> HitMatchingAlgorithm_pixelray_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_pixelray_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_window2012.h"
typedef ES_HitMatchingAlgorithm_window2012<Ref_PixelDigi_> HitMatchingAlgorithm_window2012_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_window2012_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_window2013.h"
typedef ES_HitMatchingAlgorithm_window2013<Ref_PixelDigi_> HitMatchingAlgorithm_window2013_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_window2013_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_tab2013.h"
typedef ES_HitMatchingAlgorithm_tab2013<Ref_PixelDigi_> HitMatchingAlgorithm_tab2013_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(HitMatchingAlgorithm_tab2013_PixelDigi_);

/// The Clustering Algorithms

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_a.h"
typedef ES_ClusteringAlgorithm_a<Ref_PSimHit_> ClusteringAlgorithm_a_PSimHit_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_a_PSimHit_);
typedef ES_ClusteringAlgorithm_a<Ref_PixelDigi_> ClusteringAlgorithm_a_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_a_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_broadside.h"
typedef ES_ClusteringAlgorithm_broadside<Ref_PixelDigi_> ClusteringAlgorithm_broadside_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_broadside_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_2d.h"
typedef ES_ClusteringAlgorithm_2d<Ref_PixelDigi_> ClusteringAlgorithm_2d_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_2d_PixelDigi_);

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm_neighbor.h"
typedef ES_ClusteringAlgorithm_neighbor<Ref_PixelDigi_> ClusteringAlgorithm_neighbor_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(ClusteringAlgorithm_neighbor_PixelDigi_);

/* - L1 CaloTrigger - */
//#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/L1CaloTriggerSetupProducer.h"
//DEFINE_FWK_EVENTSETUP_MODULE(L1CaloTriggerSetupProducer);


