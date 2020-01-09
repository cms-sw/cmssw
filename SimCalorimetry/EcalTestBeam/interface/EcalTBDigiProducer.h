#ifndef SimCalorimetry_EcalTestBeam_EcalTBDigiProducer_h
#define SimCalorimetry_EcalTestBeam_EcalTBDigiProducer_h

#include "FWCore/Framework/interface/ProducesCollector.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalTestBeamAlgos/interface/EcalTBReadout.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRawInfo.h"

namespace edm {
  class ConsumesCollector;
  class Event;
  class EventSetup;
  class ParameterSet;
}  // namespace edm
class PEcalTBInfo;
class PileUpEventPrincipal;

class EcalTBDigiProducer : public EcalDigiProducer {
public:
  EcalTBDigiProducer(const edm::ParameterSet &params, edm::ProducesCollector, edm::ConsumesCollector &iC);
  ~EcalTBDigiProducer() override;

  void initializeEvent(edm::Event const &, edm::EventSetup const &) override;
  void finalizeEvent(edm::Event &, edm::EventSetup const &) override;

private:
  void cacheEBDigis(const EBDigiCollection *ebDigiPtr) const override;
  void cacheEEDigis(const EEDigiCollection *eeDigiPtr) const override;

  void setPhaseShift(const DetId &detId);

  void fillTBTDCRawInfo(EcalTBTDCRawInfo &theTBTDCRawInfo);

  const EcalTrigTowerConstituentsMap m_theTTmap;
  EcalTBReadout *m_theTBReadout;

  std::string m_ecalTBInfoLabel;
  std::string m_EBdigiFinalTag;
  std::string m_EBdigiTempTag;

  bool m_doPhaseShift;
  double m_thisPhaseShift;

  bool m_doReadout;

  std::vector<EcalTBTDCRecInfoAlgo::EcalTBTDCRanges> m_tdcRanges;
  bool m_use2004OffsetConvention;

  double m_tunePhaseShift;

  mutable std::unique_ptr<EBDigiCollection> m_ebDigis;
  mutable std::unique_ptr<EEDigiCollection> m_eeDigis;
  mutable std::unique_ptr<EcalTBTDCRawInfo> m_TDCproduct;
};

#endif
