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
    ~MRHChi2MeasurementEstimatorESProducer() override;
    std::unique_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord&);

  private:
    edm::ParameterSet pset_;
  };

  MRHChi2MeasurementEstimatorESProducer::MRHChi2MeasurementEstimatorESProducer(const edm::ParameterSet& p) {
    std::string myname = p.getParameter<std::string>("ComponentName");
    pset_ = p;
    setWhatProduced(this, myname);
  }

  MRHChi2MeasurementEstimatorESProducer::~MRHChi2MeasurementEstimatorESProducer() {}

  std::unique_ptr<Chi2MeasurementEstimatorBase> MRHChi2MeasurementEstimatorESProducer::produce(
      const TrackingComponentsRecord& iRecord) {
    double maxChi2 = pset_.getParameter<double>("MaxChi2");
    double nSigma = pset_.getParameter<double>("nSigma");
    return std::make_unique<MRHChi2MeasurementEstimator>(maxChi2, nSigma);
  }

}  // namespace

DEFINE_FWK_EVENTSETUP_MODULE(MRHChi2MeasurementEstimatorESProducer);
