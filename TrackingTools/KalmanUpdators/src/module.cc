#include "TrackingTools/KalmanUpdators/interface/KFUpdatorESProducer.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorESProducer.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h" 
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(TrajectoryStateUpdator);
EVENTSETUP_DATA_REG(Chi2MeasurementEstimatorBase);
DEFINE_FWK_EVENTSETUP_MODULE(KFUpdatorESProducer);
DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(Chi2MeasurementEstimatorESProducer);

