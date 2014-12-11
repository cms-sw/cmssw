#ifndef Phase2OuterTracker_OuterTrackerStub_h
#define Phase2OuterTracker_OuterTrackerStub_h

#include <vector>
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"

class DQMStore;

class OuterTrackerStub : public edm::EDAnalyzer {

public:
  explicit OuterTrackerStub(const edm::ParameterSet&);
  ~OuterTrackerStub();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
	// TTStub stacks
	MonitorElement* Stub_Gen_Barrel = 0;
	MonitorElement* Stub_Unkn_Barrel = 0;
	MonitorElement* Stub_Comb_Barrel = 0;
	MonitorElement* Stub_Gen_Endcap = 0;
	MonitorElement* Stub_Unkn_Endcap = 0;
	MonitorElement* Stub_Comb_Endcap = 0;
	MonitorElement* Stub_Gen_Eta = 0;
	MonitorElement* Stub_Unkn_Eta = 0;
	MonitorElement* Stub_Comb_Eta = 0;
	

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
  
};
#endif
