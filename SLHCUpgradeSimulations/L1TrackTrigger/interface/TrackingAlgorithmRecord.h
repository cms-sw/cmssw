/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2011, Sept                           ///
///                                      ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGO_RECORD_H
#define TRACKING_ALGO_RECORD_H

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

  class TrackingAlgorithmRecord : public edm::eventsetup::DependentRecordImplementation< cmsUpgrades::TrackingAlgorithmRecord , boost::mpl::vector<StackedTrackerGeometryRecord , IdealMagneticFieldRecord> > {};

} /// Close namespace

#endif
