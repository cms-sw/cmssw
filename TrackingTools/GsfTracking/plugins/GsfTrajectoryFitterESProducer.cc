#include "TrackingTools/GsfTracking/plugins/GsfTrajectoryFitterESProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/GsfPropagatorWithMaterial.h"
#include "TrackingTools/GsfTracking/interface/GsfMultiStateUpdator.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateMerger.h"
#include "TrackingTools/GsfTools/interface/CloseComponentsMerger.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "TrackingTools/GsfTracking/interface/GsfChi2MeasurementEstimator.h"
#include "TrackingTools/GsfTracking/interface/GsfTrajectoryFitter.h"

#include <string>
#include <memory>

#include <iostream>

GsfTrajectoryFitterESProducer::GsfTrajectoryFitterESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

GsfTrajectoryFitterESProducer::~GsfTrajectoryFitterESProducer() {}

boost::shared_ptr<TrajectoryFitter> 
GsfTrajectoryFitterESProducer::produce(const TrajectoryFitterRecord & iRecord){ 
  //
  // material effects
  //
  std::string matName = pset_.getParameter<std::string>("MaterialEffectsUpdator");
  edm::ESHandle<GsfMaterialEffectsUpdator> matProducer;
  iRecord.getRecord<TrackingComponentsRecord>().get(matName,matProducer);
  //
  // propagator
  //
  std::string geomName = pset_.getParameter<std::string>("GeometricalPropagator");
  edm::ESHandle<Propagator> geomProducer;
  iRecord.getRecord<TrackingComponentsRecord>().get(geomName,geomProducer);
  GsfPropagatorWithMaterial propagator(*geomProducer.product(),*matProducer.product());
  //
  // merger
  //
  std::string mergerName = pset_.getParameter<std::string>("Merger");
//   edm::ESHandle<MultiTrajectoryStateMerger> mergerProducer;
//   iRecord.get(mergerName,mergerProducer);
  edm::ESHandle< MultiGaussianStateMerger<5> > mergerProducer;
  iRecord.getRecord<TrackingComponentsRecord>().get(mergerName,mergerProducer);
  MultiTrajectoryStateMerger merger(*mergerProducer.product());
  //
  // estimator
  //
  //   double chi2Cut = pset_.getParameter<double>("ChiSquarCut");
  double chi2Cut(100.);
  GsfChi2MeasurementEstimator estimator(chi2Cut);
  //
  // create algorithm
  //
  return boost::shared_ptr<TrajectoryFitter>(new GsfTrajectoryFitter(propagator,
								     GsfMultiStateUpdator(), 
								     estimator,merger));
}
