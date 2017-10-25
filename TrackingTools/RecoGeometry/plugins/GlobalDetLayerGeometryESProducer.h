#ifndef TrackingTools_RecoGeometry_GlobalDetLayerGeometryESProducer_H
#define TrackingTools_RecoGeometry_GlobalDetLayerGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/RecoGeometry/interface/GlobalDetLayerGeometry.h"
#include <memory>

class  GlobalDetLayerGeometryESProducer: public edm::ESProducer{
 public:
  GlobalDetLayerGeometryESProducer(const edm::ParameterSet & p);
  ~GlobalDetLayerGeometryESProducer() override; 
  std::shared_ptr<DetLayerGeometry> produce(const RecoGeometryRecord &);
 private:
  std::shared_ptr<DetLayerGeometry> geometry_;
};


#endif
