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

    ~CTPPSLHCInfoESSource() override {};

    edm::ESProducts<std::unique_ptr<LHCInfo>> produce(const LHCInfoRcd &);

  private:
    edm::EventRange m_validityRange;
    double m_beamEnergy;
    double m_xangle;

    bool m_insideValidityRange;

    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

CTPPSLHCInfoESSource::CTPPSLHCInfoESSource(const edm::ParameterSet& conf) :
  m_validityRange(conf.getParameter<edm::EventRange>("validityRange")),
  m_beamEnergy(conf.getParameter<double>("beamEnergy")),
  m_xangle(conf.getParameter<double>("xangle")),
  m_insideValidityRange(false)
{
  setWhatProduced(this);
  findingRecord<LHCInfoRcd>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSLHCInfoESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &key,
  const edm::IOVSyncValue& iosv, edm::ValidityInterval& oValidity)
{
  if (edm::contains(m_validityRange, iosv.eventID()))
  {
    m_insideValidityRange = true;
    oValidity = edm::ValidityInterval(edm::IOVSyncValue(m_validityRange.startEventID()), edm::IOVSyncValue(m_validityRange.endEventID()));
  } else {
    m_insideValidityRange = false;

    if (iosv.eventID() < m_validityRange.startEventID())
    {
      edm::RunNumber_t run = m_validityRange.startEventID().run();
      edm::LuminosityBlockNumber_t lb = m_validityRange.startEventID().luminosityBlock();
      edm::EventID endEvent = (lb > 1) ? edm::EventID(run, lb-1, 0) : edm::EventID(run-1, edm::EventID::maxLuminosityBlockNumber(), 0);

      oValidity = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue(endEvent));
    } else {
      edm::RunNumber_t run = m_validityRange.startEventID().run();
      edm::LuminosityBlockNumber_t lb = m_validityRange.startEventID().luminosityBlock();
      edm::EventID beginEvent = (lb < edm::EventID::maxLuminosityBlockNumber()-1) ? edm::EventID(run, lb+1, 0) : edm::EventID(run+1, 0, 0);

      oValidity = edm::ValidityInterval(edm::IOVSyncValue(beginEvent), edm::IOVSyncValue::endOfTime());
    }
  }
}

//----------------------------------------------------------------------------------------------------

edm::ESProducts<std::unique_ptr<LHCInfo>> CTPPSLHCInfoESSource::produce(const LHCInfoRcd &)
{
  auto output = std::make_unique<LHCInfo>();

  if (m_insideValidityRange)
  {
    output->setEnergy(m_beamEnergy);
    output->setCrossingAngle(m_xangle);
  } else {
    output->setEnergy(0.);
    output->setCrossingAngle(0.);
  }

  return edm::es::products(std::move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_EVENTSETUP_SOURCE(CTPPSLHCInfoESSource);
