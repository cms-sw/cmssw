#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorParams.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include <memory>

namespace {

  class Chi2MeasurementEstimatorESProducer : public edm::ESProducer {
  public:
    Chi2MeasurementEstimatorESProducer(const edm::ParameterSet& p);

    std::unique_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    const double maxChi2_;
    const double nSigma_;
    const double maxDis_;
    const double maxSag_;
    const double minTol_;
    const double minpt_;
  };

  Chi2MeasurementEstimatorESProducer::Chi2MeasurementEstimatorESProducer(const edm::ParameterSet& p)
      : maxChi2_(p.getParameter<double>("MaxChi2")),
        nSigma_(p.getParameter<double>("nSigma")),
        maxDis_(p.getParameter<double>("MaxDisplacement")),
        maxSag_(p.getParameter<double>("MaxSagitta")),
        minTol_(p.getParameter<double>("MinimalTolerance")),
        minpt_(p.getParameter<double>("MinPtForHitRecoveryInGluedDet")) {
    std::string myname = p.getParameter<std::string>("ComponentName");
    setWhatProduced(this, myname);
  }

  std::unique_ptr<Chi2MeasurementEstimatorBase> Chi2MeasurementEstimatorESProducer::produce(
      const TrackingComponentsRecord& iRecord) {
    return std::make_unique<Chi2MeasurementEstimator>(maxChi2_, nSigma_, maxDis_, maxSag_, minTol_, minpt_);
  }

  void Chi2MeasurementEstimatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    auto desc = chi2MeasurementEstimatorParams::getFilledConfigurationDescription();
    desc.add<std::string>("ComponentName", "Chi2");
    descriptions.add("Chi2MeasurementEstimatorDefault", desc);
  }

}  // namespace

DEFINE_FWK_EVENTSETUP_MODULE(Chi2MeasurementEstimatorESProducer);
