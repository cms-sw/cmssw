#include "TrackingTools/RecoGeometry/plugins/DetLayerGeometryESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>
#include <string>

using namespace edm;

DetLayerGeometryESProducer::DetLayerGeometryESProducer(const edm::ParameterSet & p) 
{
  std::string myName = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this,myName);
}
 
DetLayerGeometryESProducer::~DetLayerGeometryESProducer() {}

boost::shared_ptr<DetLayerGeometry> 
DetLayerGeometryESProducer::produce(const RecoGeometryRecord & iRecord){ 


  geometry_  = boost::shared_ptr<DetLayerGeometry>(new DetLayerGeometry());
  return geometry_;
}


DEFINE_FWK_EVENTSETUP_MODULE(DetLayerGeometryESProducer);
