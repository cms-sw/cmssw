#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilderESProducer.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

TkTransientTrackingRecHitBuilderESProducer::TkTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

TkTransientTrackingRecHitBuilderESProducer::~TkTransientTrackingRecHitBuilderESProducer() {}

boost::shared_ptr<TransientTrackingRecHitBuilder> 
TkTransientTrackingRecHitBuilderESProducer::produce(const TransientRecHitRecord & iRecord){ 
//   if (_propagator){
//     delete _propagator;
//     _propagator = 0;
//   }

  edm::ESHandle<TrackingGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );     
  
  _builder  = boost::shared_ptr<TransientTrackingRecHitBuilder>(new TkTransientTrackingRecHitBuilder(pDD.product()));
  return _builder;
}


