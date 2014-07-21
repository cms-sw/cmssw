#include "RecoTracker/SiTrackerMRHTools/plugins/SiTrackerMultiRecHitUpdatorESProducer.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

SiTrackerMultiRecHitUpdatorESProducer::SiTrackerMultiRecHitUpdatorESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

SiTrackerMultiRecHitUpdatorESProducer::~SiTrackerMultiRecHitUpdatorESProducer() {}

boost::shared_ptr<SiTrackerMultiRecHitUpdator> 
SiTrackerMultiRecHitUpdatorESProducer::produce(const MultiRecHitRecord & iRecord){ 
  std::vector<double> annealingProgram = pset_.getParameter<std::vector<double> >("AnnealingProgram");
  float Chi2Cut1D = pset_.getParameter<double>("ChiSquareCut1D");
  float Chi2Cut2D = pset_.getParameter<double>("ChiSquareCut2D");

  edm::ESHandle<TransientTrackingRecHitBuilder> hbuilder;
  std::string sname = pset_.getParameter<std::string>("TTRHBuilder");
  iRecord.getRecord<TransientRecHitRecord>().get(sname, hbuilder);
  std::string hitpropagator = pset_.getParameter<std::string>("HitPropagator");
  edm::ESHandle<TrackingRecHitPropagator> hhitpropagator;
  iRecord.getRecord<CkfComponentsRecord>().getRecord<TrackingComponentsRecord>().get(hitpropagator, hhitpropagator);		

  bool debug = pset_.getParameter<bool>("Debug");
  //_updator  = boost::shared_ptr<SiTrackerMultiRecHitUpdator>(new SiTrackerMultiRecHitUpdator(pDD.product(), pp, sp, mp, annealingProgram));
  _updator  = boost::shared_ptr<SiTrackerMultiRecHitUpdator>(new SiTrackerMultiRecHitUpdator(hbuilder.product(),hhitpropagator.product(), Chi2Cut1D, Chi2Cut2D, annealingProgram, debug));
   // _updator  = boost::shared_ptr<SiTrackerMultiRecHitUpdator>(new SiTrackerMultiRecHitUpdator(hhitpropagator.product(),annealingProgram));
  return _updator;
}


