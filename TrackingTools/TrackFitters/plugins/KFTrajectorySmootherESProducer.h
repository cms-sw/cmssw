#ifndef TrackingTools_TrackFitters_KFTrajectorySmootherESProducer_h
#define TrackingTools_TrackFitters_KFTrajectorySmootherESProducer_h

/** \class KFTrajectorySmootherESProducer
 *  ESProducer for the KFTrajectorySmoother.
 *
 *  $Date: 2007/05/09 14:17:58 $
 *  $Revision: 1.2 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include <boost/shared_ptr.hpp>

class  KFTrajectorySmootherESProducer: public edm::ESProducer{
 public:
  KFTrajectorySmootherESProducer(const edm::ParameterSet & p);
  virtual ~KFTrajectorySmootherESProducer(); 
  boost::shared_ptr<TrajectorySmoother> produce(const TrajectoryFitterRecord &);
 private:
  boost::shared_ptr<TrajectorySmoother> _smoother;
  edm::ParameterSet pset_;
};


#endif




