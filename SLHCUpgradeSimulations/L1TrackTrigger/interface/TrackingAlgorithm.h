/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2011, September                      ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGO_BASE_H
#define TRACKING_ALGO_BASE_H

#include <sstream>
#include <map>
#include <string>
#include "classNameFinder.h"

#include "SimDataFormats/SLHC/interface/L1TkTrack.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

//#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"


  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template< typename T >
  class TrackingAlgorithm
  {
    protected:
      /// Data members
      const StackedTrackerGeometry *theStackedTracker;
      std::string className_;
    public:
      /// Constructors
      TrackingAlgorithm( const StackedTrackerGeometry *aStackedGeom, std::string fName )
        : theStackedTracker( aStackedGeom ) {
	className_=classNameFinder<T>(fName);
      }

      /// Destructor
      virtual ~TrackingAlgorithm() {}

      /// Seed creation
      virtual void CreateSeeds( std::vector< L1TkTrack< T > > &output, edm::Handle< std::vector< L1TkStub< T > > > &input ) const
      {
        output.clear();
      }

      /// Match a Stub to a Seed/Track
      void AttachStubToSeed( L1TkTrack< T > &seed, edm::Ptr< L1TkStub< T > > &candidate ) const
      {
        seed.addStubPtr( candidate );
      }

/*
      virtual std::vector< L1TkTrack< T > > PropagateSeed( edm::Ptr< L1TkTracklet< T > > aSeed,
                                                                        std::vector< edm::Ptr< L1TkStub< T > > > aBricks ) const {
        std::vector< L1TkTrack< T > > emptyVector;
        emptyVector.clear();
        return emptyVector;
      }
*/

      /// Make the PSimHit equivalent
//      std::pair< double, PSimHit > MakeHit( const GeomDetUnit* dU, BaseParticlePropagator* tP, double curv ) const;
      
      /// Algorithm name
      virtual std::string AlgorithmName() const { return className_; }

  }; /// Close class



#endif

