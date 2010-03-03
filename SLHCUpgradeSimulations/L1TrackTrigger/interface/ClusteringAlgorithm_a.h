
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef CLUSTERING_ALGORITHM_a_H
#define CLUSTERING_ALGORITHM_a_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/ClusteringAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include <boost/shared_ptr.hpp>

#include <memory>
#include <string>

#include <map>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the algorithm is defined here...
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace cmsUpgrades{

template<	typename T	>
class ClusteringAlgorithm_a : public ClusteringAlgorithm<T> {
	public:
		typedef typename std::vector< T >::const_iterator inputIteratorType;

		//the member functions
		ClusteringAlgorithm_a( const cmsUpgrades::StackedTrackerGeometry *i ) : 	ClusteringAlgorithm<T>( i ), 
																					//mName(__PRETTY_FUNCTION__),
																					mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ){}
		~ClusteringAlgorithm_a() {}

		void Cluster( std::vector< std::vector< T > > &output , const std::vector< T > &input ) const {
			// no clustering, just define a "cluster" of one hit for that hit
			output.clear();
			for( inputIteratorType inputIterator = input.begin(); inputIterator != input.end(); ++inputIterator ){
				std::vector< T > temp;
				temp.push_back(*inputIterator);
				output.push_back(temp);
			}
		}

		std::string AlgorithmName() const { 
			return ( (mClassInfo->FunctionName())+"<"+(mClassInfo->TemplateTypes().begin()->second)+">" );
			//return mName;
		}

	private:
		//std::string mName;
		const cmsUpgrades::classInfo *mClassInfo;

};

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ...and declared to the framework here
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<	typename T	>
class  ES_ClusteringAlgorithm_a: public edm::ESProducer{
	public:
		ES_ClusteringAlgorithm_a(const edm::ParameterSet & p){setWhatProduced( this );}
		virtual ~ES_ClusteringAlgorithm_a() {}

		boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> > produce(const cmsUpgrades::ClusteringAlgorithmRecord & record)
		{ 
			edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
			record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
			cmsUpgrades::ClusteringAlgorithm<T>* ClusteringAlgo = new cmsUpgrades::ClusteringAlgorithm_a<T>( &(*StackedTrackerGeomHandle) );

			_theAlgo  = boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> >( ClusteringAlgo );

			return _theAlgo;
		} 

	private:
		boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> > _theAlgo;
};


#endif

