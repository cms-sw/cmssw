
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef HIT_MATCHING_ALGORITHM_a_H
#define HIT_MATCHING_ALGORITHM_a_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

//#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm_a.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"

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
class HitMatchingAlgorithm_a : public HitMatchingAlgorithm<T> {
	public:
		HitMatchingAlgorithm_a( const cmsUpgrades::StackedTrackerGeometry *i ) :	HitMatchingAlgorithm<T>( i ),
																					//mName(__PRETTY_FUNCTION__),
																					mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ) {}
		~HitMatchingAlgorithm_a() {}

		bool CheckTwoMemberHitsForCompatibility( const cmsUpgrades::LocalStub<T> & aLocalStub ) const
		{ 
			//as a test, we will accept all pairs of hits in a stack
			return true;
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
class  ES_HitMatchingAlgorithm_a: public edm::ESProducer{
	public:
		ES_HitMatchingAlgorithm_a(const edm::ParameterSet & p){setWhatProduced( this );}
		virtual ~ES_HitMatchingAlgorithm_a() {}

		boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> > produce(const cmsUpgrades::HitMatchingAlgorithmRecord & record)
		{ 
			edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
			record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
			cmsUpgrades::HitMatchingAlgorithm<T>* HitMatchingAlgo = new cmsUpgrades::HitMatchingAlgorithm_a<T>( &(*StackedTrackerGeomHandle) );

			_theAlgo  = boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> >( HitMatchingAlgo );

			return _theAlgo;
		} 

	private:
		boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> > _theAlgo;
};


#endif

