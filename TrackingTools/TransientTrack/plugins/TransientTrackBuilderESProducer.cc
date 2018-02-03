#include "TransientTrackBuilderESProducer.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "boost/mpl/vector.hpp" 
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <string>
#include <memory>

using namespace edm;

TransientTrackBuilderESProducer::TransientTrackBuilderESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

TransientTrackBuilderESProducer::~TransientTrackBuilderESProducer() {}

std::unique_ptr<TransientTrackBuilder> 
TransientTrackBuilderESProducer::produce(const TransientTrackRecord & iRecord){ 

  edm::ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get( magfield );     
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iRecord.getRecord<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 

  return std::make_unique<TransientTrackBuilder>(magfield.product(), theTrackingGeometry);

}


