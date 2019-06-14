#ifndef TrackingTools_RecoGeometry_DetLayerGeometryESProducer_H
#define TrackingTools_RecoGeometry_DetLayerGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"
#include <memory>

class DetLayerGeometryESProducer : public edm::ESProducer {
public:
  DetLayerGeometryESProducer(const edm::ParameterSet &p);
  ~DetLayerGeometryESProducer() override;
  std::unique_ptr<DetLayerGeometry> produce(const RecoGeometryRecord &);
};

#endif
