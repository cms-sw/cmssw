#ifndef TrackingTools_RecoGeometry_DetLayerGeometryESProducer_H
#define TrackingTools_RecoGeometry_DetLayerGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"
#include <memory>

class  DetLayerGeometryESProducer: public edm::ESProducer{
 public:
  DetLayerGeometryESProducer(const edm::ParameterSet & p);
  ~DetLayerGeometryESProducer() override; 
  std::shared_ptr<DetLayerGeometry> produce(const RecoGeometryRecord &);
 private:
  std::shared_ptr<DetLayerGeometry> geometry_;
};


#endif
