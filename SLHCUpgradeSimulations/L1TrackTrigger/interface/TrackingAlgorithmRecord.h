/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Nicola Pozzobon,  UNIPD              ///
///                                      ///
/// 2011, September                      ///
/// ////////////////////////////////////////

#ifndef TRACKING_ALGO_RECORD_H
#define TRACKING_ALGO_RECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "boost/mpl/vector.hpp"


  
  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  class TrackingAlgorithmRecord : public edm::eventsetup::DependentRecordImplementation< TrackingAlgorithmRecord,
                                                                                         boost::mpl::vector< StackedTrackerGeometryRecord, IdealMagneticFieldRecord> > {};



#endif

