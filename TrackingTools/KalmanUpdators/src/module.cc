#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h" 
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagator.h"
#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(TrajectoryStateUpdator);
TYPELOOKUP_DATA_REG(Chi2MeasurementEstimatorBase);
TYPELOOKUP_DATA_REG(TrackingRecHitPropagator);

