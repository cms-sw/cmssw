#ifndef TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h
#define TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h

/** \class PropagatorWithMaterialESProducer
 *  ESProducer for PropagatorWithMaterial.
 *
 *  $Date$
 *  $Revision$
 *  \author cerati
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include <boost/shared_ptr.hpp>

class  PropagatorWithMaterialESProducer: public edm::ESProducer{
 public:
  PropagatorWithMaterialESProducer(const edm::ParameterSet & p);
  virtual ~PropagatorWithMaterialESProducer(); 
  boost::shared_ptr<Propagator> produce(const TrackingComponentsRecord &);
 private:
  boost::shared_ptr<Propagator> _propagator;
  edm::ParameterSet pset_;
};


#endif




