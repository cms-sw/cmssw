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

  std::string sname = pset_.getParameter<std::string>("StripCPE");
  std::string pname = pset_.getParameter<std::string>("PixelCPE");
  
  edm::ESHandle<StripClusterParameterEstimator> se; 
  edm::ESHandle<PixelClusterParameterEstimator> pe; 
  const StripClusterParameterEstimator * sp ;
  const PixelClusterParameterEstimator * pp ;
  
  if (sname == "Fake") {
    sp = 0;
  }else{
    iRecord.getRecord<TrackerCPERecord>().get( sname, se );     
    sp = se.product();
  }
  
  if (pname == "Fake") {
    pe = 0;
  }else{
    iRecord.getRecord<TrackerCPERecord>().get( pname, pe );     
    pp = pe.product();
  }
  

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );     
  
  _builder  = boost::shared_ptr<TransientTrackingRecHitBuilder>(new TkTransientTrackingRecHitBuilder(pDD.product(), pp, sp));
  return _builder;
}


