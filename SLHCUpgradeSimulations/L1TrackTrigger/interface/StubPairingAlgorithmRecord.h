/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Sept                           ///
/// ////////////////////////////////////////

#ifndef STUB_PAIRING_ALGO_RECORD_H
#define STUB_PAIRING_ALGO_RECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "boost/mpl/vector.hpp"

namespace cmsUpgrades{
  
  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  class StubPairingAlgorithmRecord : public edm::eventsetup::DependentRecordImplementation< cmsUpgrades::StubPairingAlgorithmRecord , boost::mpl::vector<StackedTrackerGeometryRecord , IdealMagneticFieldRecord> > {};

} /// Close namespace

#endif
