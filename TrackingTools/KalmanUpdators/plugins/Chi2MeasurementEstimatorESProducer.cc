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
    ~Chi2MeasurementEstimatorESProducer() override;
    std::unique_ptr<Chi2MeasurementEstimatorBase> produce(const TrackingComponentsRecord&);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::ParameterSet const m_pset;
  };

  Chi2MeasurementEstimatorESProducer::Chi2MeasurementEstimatorESProducer(const edm::ParameterSet& p) : m_pset(p) {
    std::string myname = p.getParameter<std::string>("ComponentName");
    setWhatProduced(this, myname);
  }

  Chi2MeasurementEstimatorESProducer::~Chi2MeasurementEstimatorESProducer() {}

  std::unique_ptr<Chi2MeasurementEstimatorBase> Chi2MeasurementEstimatorESProducer::produce(
      const TrackingComponentsRecord& iRecord) {
    auto maxChi2 = m_pset.getParameter<double>("MaxChi2");
    auto nSigma = m_pset.getParameter<double>("nSigma");
    auto maxDis = m_pset.getParameter<double>("MaxDisplacement");
    auto maxSag = m_pset.getParameter<double>("MaxSagitta");
    auto minTol = m_pset.getParameter<double>("MinimalTolerance");
    auto minpt = m_pset.getParameter<double>("MinPtForHitRecoveryInGluedDet");

    return std::make_unique<Chi2MeasurementEstimator>(maxChi2, nSigma, maxDis, maxSag, minTol, minpt);
  }

  void Chi2MeasurementEstimatorESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    auto desc = chi2MeasurementEstimatorParams::getFilledConfigurationDescription();
    desc.add<std::string>("ComponentName", "Chi2");
    descriptions.add("Chi2MeasurementEstimatorDefault", desc);
  }

}  // namespace

DEFINE_FWK_EVENTSETUP_MODULE(Chi2MeasurementEstimatorESProducer);
