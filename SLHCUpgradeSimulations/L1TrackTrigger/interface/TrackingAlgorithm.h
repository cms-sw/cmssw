/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Sept                           ///
///                                      ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGO_BASE_H
#define TRACKING_ALGO_BASE_H

#include <sstream>
#include <map>

#include "SimDataFormats/SLHC/interface/L1TkTrack.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template<  typename T  >
  class TrackingAlgorithm {

    protected:
      /// Data members
      const cmsUpgrades::StackedTrackerGeometry *theStackedTracker;
      double mMagneticFieldStrength;

    public:
      /// Constructors
      TrackingAlgorithm( const cmsUpgrades::StackedTrackerGeometry *i ) : theStackedTracker(i){}
      /// Destructor
      virtual ~TrackingAlgorithm(){}

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Seed propagation
      virtual std::vector< cmsUpgrades::L1TkTrack< T > > PropagateSeed( edm::Ptr< cmsUpgrades::L1TkTracklet< T > > aSeed,
                                                                        std::vector< edm::Ptr< cmsUpgrades::L1TkStub< T > > > aBricks ) const {
        std::vector< cmsUpgrades::L1TkTrack< T > > emptyVector;
        emptyVector.clear();
        return emptyVector;
      }
      /// Make the PSimHit equivalent
      std::pair< double, PSimHit > MakeHit( const GeomDetUnit* dU, BaseParticlePropagator* tP, double curv ) const;
      
      /// Algorithm name
      virtual std::string AlgorithmName() const { return ""; }

  }; /// Close class

} /// Close namespace

#endif
