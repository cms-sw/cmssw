// framework & common header files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

// particle data table
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/ESHandle.h"

//DQM services
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <iostream>
#include <stdlib.h>

namespace edm { class HepMCProduct; }


class HiBasicGenTest : public DQMEDAnalyzer
{
 public:
  explicit HiBasicGenTest(const edm::ParameterSet&);
  virtual ~HiBasicGenTest();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c);
  void bookHistograms(DQMStore::IBooker &,
      edm::Run const &, edm::EventSetup const &) override;

 private:


  edm::EDGetTokenT<edm::HepMCProduct> generatorToken_;
  MonitorElement *dnchdeta[3];
  MonitorElement *dnchdpt[3];
  MonitorElement *b[3];
  MonitorElement *dnchdphi[3];
  MonitorElement *rp;

  edm::ESHandle < ParticleDataTable > pdt;

};
