/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
///                                      ///
/// Andrew W. Rose, IC                   ///
///                                      ///
/// 2008                                 ///
/// ////////////////////////////////////////

#ifndef CLUSTERING_ALGO_RECORD_H
#define CLUSTERING_ALGO_RECORD_H

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include "boost/mpl/vector.hpp"

  /** ************************ **/
  /**                          **/
  /**   DECLARATION OF CLASS   **/
  /**                          **/
  /** ************************ **/

  class ClusteringAlgorithmRecord : public edm::eventsetup::DependentRecordImplementation< ClusteringAlgorithmRecord,
                                                                                           boost::mpl::vector< StackedTrackerGeometryRecord, IdealMagneticFieldRecord > >{};



#endif

