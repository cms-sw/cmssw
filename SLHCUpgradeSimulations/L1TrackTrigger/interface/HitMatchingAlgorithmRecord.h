
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef HIT_MATCHING_ALGO_RECORD_H
#define HIT_MATCHING_ALGO_RECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "boost/mpl/vector.hpp"

namespace cmsUpgrades{
class HitMatchingAlgorithmRecord : public edm::eventsetup::DependentRecordImplementation< cmsUpgrades::HitMatchingAlgorithmRecord , boost::mpl::vector<StackedTrackerGeometryRecord , IdealMagneticFieldRecord> > {};
}

#endif

