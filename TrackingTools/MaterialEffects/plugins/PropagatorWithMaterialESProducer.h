#ifndef TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h
#define TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h

/** \class PropagatorWithMaterialESProducer
 *  ESProducer for PropagatorWithMaterial.
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include <memory>

class  PropagatorWithMaterialESProducer: public edm::ESProducer{
 public:
  PropagatorWithMaterialESProducer(const edm::ParameterSet & p);
  ~PropagatorWithMaterialESProducer() override; 
  std::shared_ptr<Propagator> produce(const TrackingComponentsRecord &);
 private:
  std::shared_ptr<Propagator> _propagator;
  edm::ParameterSet pset_;
};


#endif




