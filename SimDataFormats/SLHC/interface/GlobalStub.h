
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef STACKED_TRACKER_GLOBAL_STUB_FORMAT_H
#define STACKED_TRACKER_GLOBAL_STUB_FORMAT_H

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "SimDataFormats/SLHC/interface/LocalStub.h"

//#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace cmsUpgrades{

template< typename T >

class GlobalStub {

	typedef LocalStub< T > 										LocalStubType;
	//typedef std::vector	< LocalStubType >						LocalStubCollectionType;
	//typedef edm::Ref< LocalStubCollectionType , LocalStubType >	LocalStubPtrType;
	typedef edm::Ptr< LocalStubType >	LocalStubPtrType;
	
	public:

		GlobalStub()
		{
			localStubRef = LocalStubPtrType();
	  		id=0;
			globalPosition=GlobalPoint( 0.0 , 0.0 , 0.0 );
			directionVector=GlobalVector( 0.0 , 0.0 , 0.0 );
	 	}


		GlobalStub(	LocalStubPtrType aLocalStubRef )
		{
			localStubRef = aLocalStubRef;
	  		id=aLocalStubRef->Id();
			globalPosition=GlobalPoint( 0.0 , 0.0 , 0.0 );
			directionVector=GlobalVector( 0.0 , 0.0 , 0.0 );
	 	}

		GlobalStub( LocalStubPtrType aLocalStubRef , GlobalPoint aPosition , GlobalVector aVector )
		{
			localStubRef = aLocalStubRef;
			id=aLocalStubRef->Id();
			globalPosition=aPosition;
			directionVector=aVector;
		}

		GlobalStub(StackedTrackerDetId anId, GlobalPoint aPosition , GlobalVector aVector )
		{
			localStubRef = LocalStubPtrType();
			id=anId;
			globalPosition=aPosition;
			directionVector=aVector;
		}

		~GlobalStub(){}

		StackedTrackerDetId Id() const
		{
			return id;
		}

		GlobalPoint position() const
		{
			return globalPosition;
		}

		GlobalVector direction() const
		{
			return directionVector;
		}

		const LocalStubType *localStub( ) const
		{
			return &(*localStubRef);
		}

		std::string print(unsigned int i = 0 ) const {
                        std::string padding("");
                        for(unsigned int j=0;j!=i;++j)padding+="\t";

			std::stringstream output;
			output<<padding<<"GlobalStub:\n";
			padding+='\t';
			output << padding << "StackedTrackerDetId: " << id << '\n';
			output << padding << "Position: " << globalPosition << '\n';
			output << padding << "Direction: " << directionVector << '\n';
			output << localStubRef->print(i+1) ;
			return output.str();
		}


	private:
		StackedTrackerDetId 	id;
		GlobalPoint 			globalPosition;
		GlobalVector 			directionVector;
		LocalStubPtrType		localStubRef;
 
};

/*
template<	typename T	>
struct GlobalStubCollection {
	typedef std::vector		< GlobalStub< T > > 			globalStubCollection;
};

template<	typename T	>
struct GlobalStubPersistentRef {
    typedef edm::Ref		< GlobalStubCollection< T >::globalStubCollection > 	globalStubRef;
    typedef edm::RefProd	< GlobalStubCollection< T >::globalStubCollection > 	globalStubRefProd;
    typedef edm::RefVector	< GlobalStubCollection< T >::globalStubCollection > 	globalStubRefVector;
};
*/

}


template<	typename T	>
std::ostream& operator << (std::ostream& os, const cmsUpgrades::GlobalStub< T >& aGlobalStub) {
	return (os<<aGlobalStub.print() );
//	return os;
}

#endif


