// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

// particle data table
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/ESHandle.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <iostream>
#include <stdlib.h>

class HiBasicGenTest : public edm::EDAnalyzer
{
 public:
  explicit HiBasicGenTest(const edm::ParameterSet&);
  virtual ~HiBasicGenTest();
  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

 private:

  DQMStore *dbe;
  
  MonitorElement *dnchdeta[3];
  MonitorElement *dnchdpt[3];
  MonitorElement *b[3];
  MonitorElement *dnchdphi[3];
  MonitorElement *rp;

  edm::ESHandle < ParticleDataTable > pdt;

};
