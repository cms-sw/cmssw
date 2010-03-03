
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef CLUSTERING_ALGO_RECORD_H
#define CLUSTERING_ALGO_RECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "boost/mpl/vector.hpp"

namespace cmsUpgrades{
class ClusteringAlgorithmRecord : public edm::eventsetup::DependentRecordImplementation< cmsUpgrades::ClusteringAlgorithmRecord , boost::mpl::vector<StackedTrackerGeometryRecord , IdealMagneticFieldRecord> > {};
}

#endif

