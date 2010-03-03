
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef HIT_MATCHING_ALGORITHM_window_H
#define HIT_MATCHING_ALGORITHM_window_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithm.h"
#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/HitMatchingAlgorithmRecord.h"

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/WindowFinder.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h" 

#include <boost/shared_ptr.hpp>

#include <memory>
#include <string>

#include <map>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the algorithm is defined here...
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace cmsUpgrades{

template<	typename T	>
class HitMatchingAlgorithm_window : public HitMatchingAlgorithm<T> {
	public:
		HitMatchingAlgorithm_window( const cmsUpgrades::StackedTrackerGeometry *i , double aPtScalingFactor , double aIPwidth , double aRowResolution , double aColResolution  ) :
			HitMatchingAlgorithm<T>( i ),
			mWindowFinder( new cmsUpgrades::WindowFinder(i , aPtScalingFactor , aIPwidth , aRowResolution , aColResolution ) ),
																																mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) ) {}
		~HitMatchingAlgorithm_window() {}

		bool CheckTwoMemberHitsForCompatibility( const cmsUpgrades::LocalStub<T> & aLocalStub ) const;

		std::string AlgorithmName() const { 
			return ( (mClassInfo->FunctionName())+"<"+(mClassInfo->TemplateTypes().begin()->second)+">" );
		}

	private:
		cmsUpgrades::WindowFinder *mWindowFinder;
		const cmsUpgrades::classInfo *mClassInfo;

};

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// impliment the overloaded compatability test here
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<	typename T	>
bool
	cmsUpgrades::
		HitMatchingAlgorithm_window<T>::
			CheckTwoMemberHitsForCompatibility( const cmsUpgrades::LocalStub<T> & aLocalStub ) const
{
//std::cout << "==============================================================="<<std::endl ;

	typedef std::vector< T > HitCollection;
	typedef typename HitCollection::const_iterator IT;

	StackedTrackerDetId id = aLocalStub.Id();

// Calculate the average row and column for the inner hit
	double averageRow = 0.0;
	double averageCol = 0.0;

	const	HitCollection &lhits0 = aLocalStub .hit( 0 );

	if(lhits0.size()!=0){
		for (	IT hits_itr = lhits0.begin() ; hits_itr != lhits0.end() ; hits_itr++ )	{
			averageRow +=  (**hits_itr).row();
			averageCol +=  (**hits_itr).column();
		}
		averageRow /= lhits0.size();
		averageCol /= lhits0.size();
	}

//std::cout << "inner:  " << averageRow  << " , " << averageCol << " = "  ;
//mWindowFinder->dumphit( id , 0 , averageRow , averageCol  );

// Calculate window based on the average row and column of the inner hit
	cmsUpgrades::StackedTrackerWindow window = mWindowFinder->getWindow( id , averageRow , averageCol  );

//std::cout << "( " << window.mMinrow << " , " << window.mMincol << " ) & ( "<< window.mMaxrow <<" , "<< window.mMaxcol <<" ) " << std::endl;


// Calculate the average row and column for the outer hit
	averageRow = 0.0;
	averageCol = 0.0;

	const	HitCollection &lhits1 = aLocalStub .hit( 1 );

	if(lhits1.size()!=0){
		for (	IT hits_itr = lhits1.begin() ; hits_itr != lhits1.end() ; 	hits_itr++ ){
			averageRow += (**hits_itr).row();
			averageCol +=  (**hits_itr).column();
		}
		averageRow /= lhits1.size();
		averageCol /= lhits1.size();
	}

//std::cout << "outer:  " << averageRow  << " , " << averageCol << " = "  ;
//mWindowFinder->dumphit( id , 1 , averageRow , averageCol  );

	if(averageRow>=window.mMinrow){
		if(averageRow<=window.mMaxrow){
			if(averageCol>=window.mMincol){
				if(averageCol<=window.mMaxcol){
//					std::cout << "matched!"<<std::endl ;
//					std::cout << "==============================================================="<<std::endl ;
					return true;
				}
			}
		}
	}

//	std::cout << "unmatched!"<<std::endl ;
//	std::cout << "==============================================================="<<std::endl ;
	return false;
}


template<>
bool
	cmsUpgrades::
		HitMatchingAlgorithm_window<cmsUpgrades::Ref_PSimHit_>::
			CheckTwoMemberHitsForCompatibility( const cmsUpgrades::LocalStub<cmsUpgrades::Ref_PSimHit_> & aLocalStub ) const
{
	std::cout<<"simhits have no row/column info... returning false"<<std::endl;
	return false;
}










//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ...and declared to the framework here
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<	typename T	>
class  ES_HitMatchingAlgorithm_window: public edm::ESProducer{
	public:
		ES_HitMatchingAlgorithm_window(const edm::ParameterSet & p) :
			mPtThreshold( p.getParameter<double>("minPtThreshold") ),
			mIPWidth( p.getParameter<double>("ipWidth") ),
			mRowResolution( p.getParameter<double>("RowResolution") ),
			mColResolution( p.getParameter<double>("ColResolution") )
		{
			setWhatProduced( this );
		}

		virtual ~ES_HitMatchingAlgorithm_window() {}

		boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> > produce(const cmsUpgrades::HitMatchingAlgorithmRecord & record)
		{ 
			edm::ESHandle<MagneticField> magnet;
			record.getRecord<IdealMagneticFieldRecord>().get(magnet);
			double mMagneticFieldStrength = magnet->inTesla(GlobalPoint(0,0,0)).z();

			double mPtScalingFactor = 0.0015*mMagneticFieldStrength/mPtThreshold;

			edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
			record.getRecord<cmsUpgrades::StackedTrackerGeometryRecord>().get( StackedTrackerGeomHandle );
  
			cmsUpgrades::HitMatchingAlgorithm<T>* HitMatchingAlgo = new cmsUpgrades::HitMatchingAlgorithm_window<T>( &(*StackedTrackerGeomHandle), mPtScalingFactor , mIPWidth , mRowResolution , mColResolution );

			_theAlgo  = boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> >( HitMatchingAlgo );

			return _theAlgo;
		} 

	private:
		boost::shared_ptr< cmsUpgrades::HitMatchingAlgorithm<T> > _theAlgo;
		double mPtThreshold;
		double mIPWidth;
		double mRowResolution;
		double mColResolution;
};


#endif

