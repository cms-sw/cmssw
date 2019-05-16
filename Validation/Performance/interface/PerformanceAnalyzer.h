#ifndef PerformanceAnalyzer_H
#define PerformanceAnalyzer_H

// user include files

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

namespace edm {class EventTime;}

class PerformanceAnalyzer : public DQMEDAnalyzer
{

public:
  explicit PerformanceAnalyzer(const edm::ParameterSet&);
  ~PerformanceAnalyzer() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<edm::EventTime> eventTime_Token_;
  std::string              fOutputFile ;
  MonitorElement*          fVtxSmeared ;
  MonitorElement*          fg4SimHits ;
  MonitorElement*          fMixing ;
  MonitorElement*          fSiPixelDigis ;
  MonitorElement*          fSiStripDigis ;
  MonitorElement*          fEcalUnsuppDigis ;
  MonitorElement*          fEcalZeroSuppDigis ;
  MonitorElement*          fPreShwZeroSuppDigis ;
  MonitorElement*          fHcalUnsuppDigis ;
  MonitorElement*          fMuonCSCDigis ;
  MonitorElement*          fMuonDTDigis ;
  MonitorElement*          fMuonRPCDigis ;

};

#endif

