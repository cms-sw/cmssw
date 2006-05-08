#ifndef TrackingTools_GenericTransientTrackingRecHitBuilderESProducer_h
#define TrackingTools_GenericTransientTrackingRecHitBuilderESProducer_h

/** \class GenericTransientTrackingRecHitBuilderESProducer
 *  ESProducer for GenericTransientTrackingRecHitBuilder
 *
 *  $Date: $
 *  $Revision:$
 *  \author Chang Liu - Purdue University
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include <boost/shared_ptr.hpp>

class  GenericTransientTrackingRecHitBuilderESProducer: public edm::ESProducer{
 public:
  GenericTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet & p);
  virtual ~GenericTransientTrackingRecHitBuilderESProducer(); 
  boost::shared_ptr<TransientTrackingRecHitBuilder> produce(const TransientRecHitRecord &);
 private:
  boost::shared_ptr<TransientTrackingRecHitBuilder> theBuilder;
  edm::ParameterSet thePSet;
};

#endif

