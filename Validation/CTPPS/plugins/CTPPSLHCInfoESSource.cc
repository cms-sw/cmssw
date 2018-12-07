// Original Author:  Jan Kašpar

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducts.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Provides LHCInfo data necessary for CTPPS reconstruction (and direct simulation).
 **/
class CTPPSLHCInfoESSource: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder
{
  public:
    CTPPSLHCInfoESSource(const edm::ParameterSet &);

    ~CTPPSLHCInfoESSource() {};

    edm::ESProducts<std::unique_ptr<LHCInfo>> produce(const LHCInfoRcd &);

  private:
    double m_beamEnergy;
    double m_xangle;

    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoESSource::CTPPSLHCInfoESSource(const edm::ParameterSet& conf) :
  m_beamEnergy(conf.getParameter<double>("beamEnergy")),
  m_xangle(conf.getParameter<double>("xangle"))
{
  setWhatProduced(this);
  findingRecord<LHCInfoRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
  const edm::IOVSyncValue& iosv, edm::ValidityInterval& oValidity)
{
  oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime());
}

//----------------------------------------------------------------------------------------------------

edm::ESProducts<std::unique_ptr<LHCInfo>> CTPPSLHCInfoESSource::produce(const LHCInfoRcd &)
{
  auto output = std::make_unique<LHCInfo>();

  output->setEnergy(m_beamEnergy);
  output->setCrossingAngle(m_xangle);

  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSLHCInfoESSource);
