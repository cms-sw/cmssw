/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Oct                            ///
///                                      ///
/// ////////////////////////////////////////

#ifndef STUB_PAIRING_ALGO_BASE_H
#define STUB_PAIRING_ALGO_BASE_H

#include <sstream>
#include <map>

#include "SimDataFormats/SLHC/interface/L1TkStub.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

namespace cmsUpgrades{

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  template<  typename T  >
  class StubPairingAlgorithm {

    protected:
      /// Data members
      const cmsUpgrades::StackedTrackerGeometry *theStackedTracker;

    public:
      /// Constructors
      StubPairingAlgorithm( const cmsUpgrades::StackedTrackerGeometry *i ) : theStackedTracker(i){}
      /// Destructor
      virtual ~StubPairingAlgorithm(){}

      /// ////////////// ///
      /// HELPER METHODS ///
      /// Matching operations
      virtual bool  CheckTwoStackStubsForCompatibility( edm::Ptr< L1TkStub< T > > innerStub, edm::Ptr< L1TkStub< T > > outerStub ) const {
        return false;
      }

      /// Algorithm name
      virtual std::string AlgorithmName() const { return ""; }

  }; /// Close class

} /// Close namespace

#endif

