#ifndef TrackingTools_TrackFitters_KFTrajectoryFitterESProducer_h
#define TrackingTools_TrackFitters_KFTrajectoryFitterESProducer_h

/** \class KFTrajectoryFitterESProducer
 *  ESProducer for the KFTrajectoryFitter.
 *
 *  $Date: 2007/05/09 14:17:58 $
 *  $Revision: 1.2 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitterRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include <boost/shared_ptr.hpp>

class  KFTrajectoryFitterESProducer: public edm::ESProducer{
 public:
  KFTrajectoryFitterESProducer(const edm::ParameterSet & p);
  virtual ~KFTrajectoryFitterESProducer(); 
  boost::shared_ptr<TrajectoryFitter> produce(const TrajectoryFitterRecord &);
 private:
  boost::shared_ptr<TrajectoryFitter> _fitter;
  edm::ParameterSet pset_;
};


#endif




