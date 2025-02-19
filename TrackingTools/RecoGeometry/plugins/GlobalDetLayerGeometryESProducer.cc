#include "TrackingTools/RecoGeometry/plugins/GlobalDetLayerGeometryESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>
#include <string>

using namespace edm;

GlobalDetLayerGeometryESProducer::GlobalDetLayerGeometryESProducer(const edm::ParameterSet & p) 
{
  std::string myName = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this,myName);
}
 
GlobalDetLayerGeometryESProducer::~GlobalDetLayerGeometryESProducer() {}

boost::shared_ptr<DetLayerGeometry> 
GlobalDetLayerGeometryESProducer::produce(const RecoGeometryRecord & iRecord){ 
  
  edm::ESHandle<GeometricSearchTracker> tracker;  
  edm::ESHandle<MuonDetLayerGeometry> muon;

  iRecord.getRecord<TrackerRecoGeometryRecord>().get(tracker);
  iRecord.getRecord<MuonRecoGeometryRecord>().get(muon);

  geometry_  = boost::shared_ptr<DetLayerGeometry>(new GlobalDetLayerGeometry(tracker.product(),
									      muon.product() ));
  return geometry_;
}


DEFINE_FWK_EVENTSETUP_MODULE(GlobalDetLayerGeometryESProducer);
