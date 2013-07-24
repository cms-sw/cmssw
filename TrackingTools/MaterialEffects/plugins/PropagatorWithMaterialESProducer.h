#ifndef TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h
#define TrackingTools_ESProducers_PropagatorWithMaterialESProducer_h

/** \class PropagatorWithMaterialESProducer
 *  ESProducer for PropagatorWithMaterial.
 *
 *  $Date: 2007/05/09 14:11:36 $
 *  $Revision: 1.2 $
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




