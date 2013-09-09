/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011 June                            ///
///                                      ///
/// Removed (NOT commented) TTHits       ///
/// (Maybe in the future they will be    ///
/// reintroduced in the framework...)    ///
/// ////////////////////////////////////////

#include "FWCore/Utilities/interface/typelookup.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"

TYPELOOKUP_DATA_REG(ClusteringAlgorithm<Ref_PSimHit_>);
TYPELOOKUP_DATA_REG(ClusteringAlgorithm<Ref_PixelDigi_>);

