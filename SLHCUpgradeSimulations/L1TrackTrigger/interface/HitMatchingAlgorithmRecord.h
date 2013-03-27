/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
/// Nicola Pozzobon, UNIPD               ///
///                                      ///
/// 2008                                 ///
/// 2010, May                            ///
/// ////////////////////////////////////////

#ifndef HIT_MATCHING_ALGO_RECORD_H
#define HIT_MATCHING_ALGO_RECORD_H

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

  class HitMatchingAlgorithmRecord : public edm::eventsetup::DependentRecordImplementation< HitMatchingAlgorithmRecord,
                                                                                            boost::mpl::vector< StackedTrackerGeometryRecord, IdealMagneticFieldRecord > >{};



#endif

