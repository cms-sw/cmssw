#ifndef TrackingTools_ESProducers_KFUpdatorESProducer_h
#define TrackingTools_ESProducers_KFUpdatorESProducer_h

/** KFUpdatorESProducer
 *  ESProducer for KFUpdator class.
 *
 *  $Date: 2007/05/09 13:50:25 $
 *  $Revision: 1.2 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include <boost/shared_ptr.hpp>

class  KFUpdatorESProducer: public edm::ESProducer{
 public:
  KFUpdatorESProducer(const edm::ParameterSet & p);
  virtual ~KFUpdatorESProducer(); 
  boost::shared_ptr<TrajectoryStateUpdator> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<TrajectoryStateUpdator> _updator;
  edm::ParameterSet pset_;
};


#endif




