
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef HIT_MATCHING_ALGO_BASE_H
#define HIT_MATCHING_ALGO_BASE_H

#include "SimDataFormats/SLHC/interface/LocalStub.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include <sstream>
#include <map>

//using namespace std;

namespace cmsUpgrades{

template<	typename T	>
class HitMatchingAlgorithm {
	public:

		HitMatchingAlgorithm( const cmsUpgrades::StackedTrackerGeometry *i ) : theStackedTracker(i){}

		virtual ~HitMatchingAlgorithm(){}

		virtual bool CheckTwoMemberHitsForCompatibility( const cmsUpgrades::LocalStub<T> &aLocalStub ) const {
			return false;
		}

		virtual std::string AlgorithmName() const { return ""; }


	protected:
		const cmsUpgrades::StackedTrackerGeometry *theStackedTracker;
};


}

#endif

