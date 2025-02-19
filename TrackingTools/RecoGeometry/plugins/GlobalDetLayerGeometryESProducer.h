#ifndef TrackingTools_RecoGeometry_GlobalDetLayerGeometryESProducer_H
#define TrackingTools_RecoGeometry_GlobalDetLayerGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"
#include "TrackingTools/RecoGeometry/interface/GlobalDetLayerGeometry.h"
#include <boost/shared_ptr.hpp>

class  GlobalDetLayerGeometryESProducer: public edm::ESProducer{
 public:
  GlobalDetLayerGeometryESProducer(const edm::ParameterSet & p);
  virtual ~GlobalDetLayerGeometryESProducer(); 
  boost::shared_ptr<DetLayerGeometry> produce(const RecoGeometryRecord &);
 private:
  boost::shared_ptr<DetLayerGeometry> geometry_;
};


#endif
