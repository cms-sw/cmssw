/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, June                           ///
///                                      ///
/// Added feature:                       ///
/// Possibility to have Fake Stub flag   ///
/// in Simulations 'isFake()' and Trk ID ///
/// too 'trackID()'. A Stub is flagged   ///
/// as Fake in the StubBuilder if the    ///
/// hits come from different SimTracks.  ///
/// More details in the StubBuilder      ///
/// files, including about templates for ///
/// different kind of Stubs              ///
/// ////////////////////////////////////////

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

  typedef LocalStub< T >                     LocalStubType;
  typedef edm::Ptr< LocalStubType >  LocalStubPtrType;
  
  public:

    GlobalStub()
    {
      genuinity = true;
      trkid = -9999;
      localStubRef = LocalStubPtrType();
        id=0;
      globalPosition=GlobalPoint( 0.0 , 0.0 , 0.0 );
      directionVector=GlobalVector( 0.0 , 0.0 , 0.0 );
     }


    GlobalStub(  LocalStubPtrType aLocalStubRef )
    {
      trkid = -9999;
      genuinity = false;
      localStubRef = aLocalStubRef;
        id=aLocalStubRef->Id();
      globalPosition=GlobalPoint( 0.0 , 0.0 , 0.0 );
      directionVector=GlobalVector( 0.0 , 0.0 , 0.0 );
     }

    GlobalStub( LocalStubPtrType aLocalStubRef , GlobalPoint aPosition , GlobalVector aVector )
    {
      trkid = -9999;
      genuinity = false;
      localStubRef = aLocalStubRef;
      id=aLocalStubRef->Id();
      globalPosition=aPosition;
      directionVector=aVector;
    }

    GlobalStub(StackedTrackerDetId anId, GlobalPoint aPosition , GlobalVector aVector )
    {
      trkid = -9999;
      genuinity = false;
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

    /// New for 3_3_6
    /// Genuinity and Track ID
    bool isFake() const {
      if (genuinity == true) return false;
      else return true;
    }
    void setGenuinity (bool a) {
      genuinity = a;
    }
    int trackID() const {
      return trkid;
    }
    void setTrackID( int a ) {
      trkid = a;
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
    bool genuinity;
    int trkid;
    StackedTrackerDetId   id;
    GlobalPoint       globalPosition;
    GlobalVector       directionVector;
    LocalStubPtrType    localStubRef;
 
};


}


template<  typename T  >
std::ostream& operator << (std::ostream& os, const cmsUpgrades::GlobalStub< T >& aGlobalStub) {
  return (os<<aGlobalStub.print() );
//  return os;
}

#endif


