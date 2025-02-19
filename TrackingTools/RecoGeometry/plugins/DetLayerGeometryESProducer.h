#ifndef TrackingTools_RecoGeometry_DetLayerGeometryESProducer_H
#define TrackingTools_RecoGeometry_DetLayerGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"
#include <boost/shared_ptr.hpp>

class  DetLayerGeometryESProducer: public edm::ESProducer{
 public:
  DetLayerGeometryESProducer(const edm::ParameterSet & p);
  virtual ~DetLayerGeometryESProducer(); 
  boost::shared_ptr<DetLayerGeometry> produce(const RecoGeometryRecord &);
 private:
  boost::shared_ptr<DetLayerGeometry> geometry_;
};


#endif
