/** \class GenericTransientTrackingRecHitBuilderESProducer
 *  ESProducer for GenericTransientTrackingRecHitBuilder
 *
 *  $Date: $
 *  $Revision:$
 *  \author Chang Liu - Purdue University
 */

#include "TrackingTools/TransientTrackingRecHit/src/GenericTransientTrackingRecHitBuilderESProducer.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHitBuilder.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

GenericTransientTrackingRecHitBuilderESProducer::GenericTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  thePSet = p;
  setWhatProduced(this,myname);
}

GenericTransientTrackingRecHitBuilderESProducer::~GenericTransientTrackingRecHitBuilderESProducer() {}

boost::shared_ptr<TransientTrackingRecHitBuilder> 
GenericTransientTrackingRecHitBuilderESProducer::produce(const TransientRecHitRecord & iRecord){ 

  edm::ESHandle<TrackingGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(pDD);     
  
  theBuilder  = boost::shared_ptr<TransientTrackingRecHitBuilder>(new GenericTransientTrackingRecHitBuilder(pDD.product()));
  return theBuilder;
}

