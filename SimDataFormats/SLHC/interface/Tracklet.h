
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef TRACKLET_H
#define TRACKLET_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "SimDataFormats/SLHC/interface/GlobalStub.h"
//#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

//for the helper methods
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

namespace cmsUpgrades{

template< typename T >
class Tracklet {

		typedef GlobalStub< T > 										GlobalStubType;
		//typedef std::vector	< GlobalStubType >							GlobalStubCollectionType;
		//typedef edm::Ref< GlobalStubCollectionType, GlobalStubType >	GlobalStubPtrType;
		typedef edm::Ptr< GlobalStubType >	GlobalStubPtrType;


public:
		typedef	std::set< std::pair<unsigned int , GlobalStubPtrType> >		TrackletMap;
		typedef typename TrackletMap::const_iterator 											TrackletMapIterator;

//		typedef	std::map< unsigned int , GlobalStubPtrType >		TrackletMap;
//		typedef	std::map< unsigned int , PTrajectoryStateOnDet >	TrajectoryMap;

		Tracklet()
		{
			theStubs.clear();
			theVertex=GlobalPoint(0.0,0.0,0.0);
//			theTrajectories.clear();
			NullStub=GlobalStubPtrType();
 	 	}


		virtual ~Tracklet(){}

		void addHit(	unsigned int aStubIdentifier,
						const GlobalStubPtrType &aHit )
		{
	    	theStubs.insert( std::make_pair( aStubIdentifier , aHit ) ); 
		}

/*		void addTrajectory( unsigned int aStubIdentifier , const PTrajectoryStateOnDet& trajectory )
		{
			theTrajectories.insert( std::make_pair( aStubIdentifier , trajectory ) ); 	
		}*/

		void addVertex( const GlobalPoint & aVertex )
		{
			theVertex=aVertex;
		}

		const GlobalStubPtrType &stub( unsigned int aStubIdentifier ) const
		{
			for (TrackletMapIterator i = theStubs.begin(); i != theStubs.end(); ++i){
				if ( i->first == aStubIdentifier ) return i->second;
			}
			return NullStub;

			/*if( theStubs.find(aStubIdentifier) != theStubs.end()  ){
				return theStubs.find(aStubIdentifier)->second;
			}else{
				return NullStub;
			}*/
		}

		const TrackletMap& stubs() const
		{
			return theStubs;
		}

		const GlobalPoint& vertex()
		{
			return theVertex;
		}

		//useful methods
		FastHelix HelixFit(const edm::EventSetup& iSetup)
		{
			return	FastHelix(	theStubs.rbegin()->second->position(),
								theStubs.begin()->second->position(),
								theVertex,
								iSetup
							);
		}
	
		FreeTrajectoryState VertexTrajectoryState(const edm::EventSetup& iSetup)
		{
			AlgebraicSymMatrix errorMatrix(5,1);
			CurvilinearTrajectoryError initialError(errorMatrix*100.);
			return	FreeTrajectoryState(this->HelixFit(iSetup).stateAtVertex().parameters(),initialError);
		}

		double twoPointPt() const
		{
			GlobalPoint inner = theStubs.begin()->second->position();
			GlobalPoint outer = theStubs.rbegin()->second->position();
			double phidiff = outer.phi() - inner.phi();
			double r1 = inner.perp()/100;
			double r2 = outer.perp()/100;
			double x2 = r1*r1 + r2*r2 - 2*r1*r2*cos(phidiff);
			return 0.6*sqrt(x2)/sin(fabs(phidiff));
			//return 0.0;
		}

		std::string print(unsigned int i = 0 ) const {
                        std::string padding("");
                        for(unsigned int j=0;j!=i;++j)padding+="\t";

			std::stringstream output;
			output<<padding<<"Tracklet:\n";
			padding+='\t';
			output << padding << "Projected Vertex: " << theVertex << '\n';
			output << padding << "Two Point Pt: " << this->twoPointPt() << '\n';

			for( TrackletMapIterator it=theStubs.begin() ; it!=theStubs.end() ; ++it )
				output << it->second->print(i+1) << '\n';

			return output.str();
		}



	private:
	//which hits formed the tracklet
		TrackletMap theStubs;
		GlobalPoint theVertex;
//		TrajectoryMap theTrajectories;

		GlobalStubPtrType NullStub;

};

}

template<	typename T	>
std::ostream& operator << (std::ostream& os, const cmsUpgrades::Tracklet< T >& aTracklet) {
	return (os<<aTracklet.print() );
//	return os;
}

#endif

