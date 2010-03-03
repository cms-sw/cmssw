
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef CLUSTERING_ALGORITHM_broadside_H
#define CLUSTERING_ALGORITHM_broadside_H

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
class ClusteringAlgorithm_broadside : public ClusteringAlgorithm<T> {
	public:
		typedef typename std::vector< T >::const_iterator inputIteratorType;

		//the member functions
		ClusteringAlgorithm_broadside( const cmsUpgrades::StackedTrackerGeometry *i , int aWidthCut) : 	ClusteringAlgorithm<T>( i ), 
																					mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ),
																					mWidthCut(aWidthCut){}
		~ClusteringAlgorithm_broadside() {}

		void Cluster( std::vector< std::vector< T > > &output , const std::vector< T > &input ) const;

		std::string AlgorithmName() const { 
			return ( (mClassInfo->FunctionName())+"<"+(mClassInfo->TemplateTypes().begin()->second)+">" );
		}

	private:
		const cmsUpgrades::classInfo *mClassInfo;
		int mWidthCut;
};


template<typename T>
void ClusteringAlgorithm_broadside< T >::Cluster( std::vector< std::vector< T > > &output , const std::vector< T > &input ) const {
	output.clear();

	std::map<unsigned int , std::vector<T> > local;
	typedef typename std::map<unsigned int , std::vector<T> >::iterator mapIteratorType;

	typedef typename std::vector< T >::const_iterator inputIteratorType;
	typedef typename std::vector< T >::iterator internalIteratorType;

	inputIteratorType inputIterator = input.begin();
	while( inputIterator != input.end() ){
		local[(**inputIterator).column()].push_back(*inputIterator);
		++inputIterator;
	}

	mapIteratorType mapIterator = local.begin();
	while( mapIterator != local.end() ){

		internalIteratorType inputIterator = mapIterator->second.begin();
		while( inputIterator != mapIterator->second.end() ){
			std::vector< T > temp;
			temp.push_back(*inputIterator);
			inputIterator=mapIterator->second.erase(inputIterator);
			internalIteratorType inputIterator2 = inputIterator;

			while( inputIterator2 != mapIterator->second.end() ){
				if( (temp.back()->column() == (**inputIterator2).column()) && ((**inputIterator2).row() - temp.back()->row() == 1) ){
					temp.push_back(*inputIterator2);
					inputIterator2=mapIterator->second.erase(inputIterator2);
				}else{
					break;
				}
			}
			if ( (mWidthCut<1) || (int(temp.size())<=mWidthCut) ) output.push_back(temp);
			inputIterator=inputIterator2;		
		}
		++mapIterator;
	}

}

//For simhits, no clustering, just define a "cluster" of one hit for that hit
template<>
void ClusteringAlgorithm_broadside<cmsUpgrades::Ref_PSimHit_>::Cluster( std::vector< std::vector<cmsUpgrades::Ref_PSimHit_> > &output , const std::vector<cmsUpgrades::Ref_PSimHit_> &input ) const {
	output.clear();
	std::vector< cmsUpgrades::Ref_PSimHit_ >::const_iterator inputIterator;
	for( inputIterator = input.begin(); inputIterator != input.end(); ++inputIterator ){
		std::vector< cmsUpgrades::Ref_PSimHit_ > temp;
		temp.push_back(*inputIterator);
		output.push_back(temp);
	}
}



}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ...and declared to the framework here
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<	typename T	>
class  ES_ClusteringAlgorithm_broadside: public edm::ESProducer{
	public:
		ES_ClusteringAlgorithm_broadside(const edm::ParameterSet & p) : mWidthCut( p.getParameter<int>("WidthCut") ) {setWhatProduced( this );}
		virtual ~ES_ClusteringAlgorithm_broadside() {}

		boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> > produce(const cmsUpgrades::ClusteringAlgorithmRecord & record)
		{ 
			edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
			record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
			cmsUpgrades::ClusteringAlgorithm<T>* ClusteringAlgo = new cmsUpgrades::ClusteringAlgorithm_broadside<T>( &(*StackedTrackerGeomHandle) , mWidthCut );

			_theAlgo  = boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> >( ClusteringAlgo );

			return _theAlgo;
		} 

	private:
		boost::shared_ptr< cmsUpgrades::ClusteringAlgorithm<T> > _theAlgo;
		int mWidthCut;
};


#endif

