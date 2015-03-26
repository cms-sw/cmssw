#ifndef TrackPropagators_ESProducers_GeantPropagatorESProducer_h
#define TrackPropagators_ESProducers_GeantPropagatorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include <boost/shared_ptr.hpp>

/*
 * GeantPropagatorESProducer
 *
 * Produces an Geant4ePropagator for track propagation
 *
 */

class GeantPropagatorESProducer: public edm::ESProducer{
 public:
  GeantPropagatorESProducer(const edm::ParameterSet & p);
  virtual ~GeantPropagatorESProducer() override; 

  boost::shared_ptr<Propagator> produce(const TrackingComponentsRecord &);

 private:
  boost::shared_ptr<Propagator> _propagator;
  edm::ParameterSet pset_;
};


#endif




