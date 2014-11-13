#ifndef Phase2OuterTracker_OuterTrackerCluster_h
#define Phase2OuterTracker_OuterTrackerCluster_h

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

class OuterTrackerCluster : public edm::EDAnalyzer {

public:
  explicit OuterTrackerCluster(const edm::ParameterSet&);
  ~OuterTrackerCluster();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
	// TTCluster stacks
	MonitorElement* Cluster_Gen_Barrel = 0;
	MonitorElement* Cluster_Unkn_Barrel = 0;
	MonitorElement* Cluster_Comb_Barrel = 0;
	MonitorElement* Cluster_Gen_Endcap = 0;
	MonitorElement* Cluster_Unkn_Endcap = 0;
	MonitorElement* Cluster_Comb_Endcap = 0;
	MonitorElement* Cluster_Gen_Eta = 0;
	MonitorElement* Cluster_Unkn_Eta = 0;
	MonitorElement* Cluster_Comb_Eta = 0;
	

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
  
};
#endif
