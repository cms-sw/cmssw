#ifndef TrackingTools_ESProducers_KFSwitching1DUpdatorESProducer_h
#define TrackingTools_ESProducers_KFSwitching1DUpdatorESProducer_h

/** KFSwitching1DUpdatorESProducer
 *  ESProducer for KFSwitching1DUpdator class.
 *
 *  $Date: 2009/09/09 15:40:39 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdator.h"
#include <boost/shared_ptr.hpp>

class  KFSwitching1DUpdatorESProducer: public edm::ESProducer{
 public:
  KFSwitching1DUpdatorESProducer(const edm::ParameterSet & p);
  virtual ~KFSwitching1DUpdatorESProducer(); 
  boost::shared_ptr<TrajectoryStateUpdator> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<TrajectoryStateUpdator> _updator;
  edm::ParameterSet pset_;
};


#endif




