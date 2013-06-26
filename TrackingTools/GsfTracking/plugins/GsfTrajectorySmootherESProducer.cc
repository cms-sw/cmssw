#include "TrackingTools/GsfTracking/plugins/GsfTrajectorySmootherESProducer.h"

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
#include "TrackingTools/GsfTracking/interface/GsfTrajectorySmoother.h"

#include <string>
#include <memory>

GsfTrajectorySmootherESProducer::GsfTrajectorySmootherESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myname);
}

GsfTrajectorySmootherESProducer::~GsfTrajectorySmootherESProducer() {}

boost::shared_ptr<TrajectorySmoother> 
GsfTrajectorySmootherESProducer::produce(const TrajectoryFitterRecord & iRecord){ 
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
  // geometry
  std::string gname = pset_.getParameter<std::string>("RecoGeometry");
  edm::ESHandle<DetLayerGeometry> geo;
  iRecord.getRecord<RecoGeometryRecord>().get(gname,geo);
  // create algorithm
  //
  //   bool matBefUpd = pset_.getParameter<bool>("MaterialBeforeUpdate");
  double scale = pset_.getParameter<double>("ErrorRescaling");
  return boost::shared_ptr<TrajectorySmoother>(new GsfTrajectorySmoother(propagator,
									 GsfMultiStateUpdator(), 
									 estimator,merger,
// 									 matBefUpd,
									 scale,
									 true,//BM should this be taken from parameterSet?
									 geo.product()));
}
