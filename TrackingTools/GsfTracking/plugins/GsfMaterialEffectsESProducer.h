#ifndef GsfMaterialEffectsESProducer_h_
#define GsfMaterialEffectsESProducer_h_

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include <boost/shared_ptr.hpp>

/** Provides algorithms for estimating material effects (GSF compatible).
 * Multiple scattering estimates can be provided according to a single (== "KF") 
 * or two-component model. Energy loss estimates can be provided according to 
 * a single component ionization- or radiation model (== "KF") or a multi-component
 * Bethe-Heitler model. */

class  GsfMaterialEffectsESProducer: public edm::ESProducer{
 public:
  GsfMaterialEffectsESProducer(const edm::ParameterSet & p);
  virtual ~GsfMaterialEffectsESProducer(); 
  boost::shared_ptr<GsfMaterialEffectsUpdator> produce(const TrackingComponentsRecord &);
 private:
  edm::ParameterSet pset_;
};


#endif




