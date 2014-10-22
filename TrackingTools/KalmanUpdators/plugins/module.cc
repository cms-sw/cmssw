#include "TrackingTools/KalmanUpdators/interface/KFUpdatorESProducer.h"
#include "TrackingTools/KalmanUpdators/interface/KFSwitching1DUpdatorESProducer.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorESProducer.h"
#include "TrackingTools/KalmanUpdators/interface/MRHChi2MeasurementEstimatorESProducer.h"
#include "TrackingTools/KalmanUpdators/interface/TrackingRecHitPropagatorESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

DEFINE_FWK_EVENTSETUP_MODULE(KFUpdatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(KFSwitching1DUpdatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(Chi2MeasurementEstimatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(MRHChi2MeasurementEstimatorESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(TrackingRecHitPropagatorESProducer);

