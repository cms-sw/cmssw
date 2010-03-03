
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef CLUSTERING_ALGO_BASE_H
#define CLUSTERING_ALGO_BASE_H

#include "SimDataFormats/SLHC/interface/LocalStub.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include <sstream>
#include <map>

//using namespace std;

namespace cmsUpgrades{

template<	typename T	>
class ClusteringAlgorithm {
	public:
		ClusteringAlgorithm( const cmsUpgrades::StackedTrackerGeometry *i ) : theStackedTracker(i){}

		virtual ~ClusteringAlgorithm(){}

		virtual void Cluster( std::vector< std::vector< T > > &output , const std::vector< T > &input ) const {
			output.clear();
		}

		virtual std::string AlgorithmName() const { return ""; }


	protected:
		const cmsUpgrades::StackedTrackerGeometry *theStackedTracker;
};


}

#endif

