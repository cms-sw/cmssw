#ifndef TrackingToolsKalmanUpdatorsChi2MeasurementEstimatorParams_H
#define TrackingToolsKalmanUpdatorsChi2MeasurementEstimatorParams_H
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
namespace chi2MeasurementEstimatorParams {
  inline edm::ParameterSetDescription getFilledConfigurationDescription() {
    edm::ParameterSetDescription desc;
    desc.add<double>("MaxChi2", 30);
    desc.add<double>("nSigma", 3);
    desc.add<double>("MaxDisplacement", 0.5);
    desc.add<double>("MaxSagitta", 2.);
    desc.add<double>("MinimalTolerance", 0.5);
    desc.add<double>("MinPtForHitRecoveryInGluedDet", 1.e12);  // for mitigation use  0.9);
    return desc;
  }
}  // namespace chi2MeasurementEstimatorParams
#endif
