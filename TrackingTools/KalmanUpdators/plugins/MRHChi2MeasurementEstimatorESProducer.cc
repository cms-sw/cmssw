#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/MRHChi2MeasurementEstimator.h"
#include <memory>

namespace {

  class MRHChi2MeasurementEstimatorESProducer : public edm::ESProducer {
  public:
    MRHChi2MeasurementEstimatorESProducer(const edm::ParameterSet& p);

    std::unique_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    const double maxChi2_;
    const double nSigma_;
  };

  MRHChi2MeasurementEstimatorESProducer::MRHChi2MeasurementEstimatorESProducer(const edm::ParameterSet& p)
      : maxChi2_(p.getParameter<double>("MaxChi2")), nSigma_(p.getParameter<double>("nSigma")) {
    std::string myname = p.getParameter<std::string>("ComponentName");
    setWhatProduced(this, myname);
  }

  std::unique_ptr<Chi2MeasurementEstimatorBase> MRHChi2MeasurementEstimatorESProducer::produce(
      const TrackingComponentsRecord& iRecord) {
    return std::make_unique<MRHChi2MeasurementEstimator>(maxChi2_, nSigma_);
  }

  void MRHChi2MeasurementEstimatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<double>("MaxChi2");
    desc.add<double>("nSigma");
    desc.add<std::string>("ComponentName");
    descriptions.addDefault(desc);
  }
}  // namespace

DEFINE_FWK_EVENTSETUP_MODULE(MRHChi2MeasurementEstimatorESProducer);
