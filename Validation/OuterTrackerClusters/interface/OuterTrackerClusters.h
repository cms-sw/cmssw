#ifndef OuterTrackerClusters_OuterTrackerClusters_h
#define OuterTrackerClusters_OuterTrackerClusters_h

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMStore.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"

#include <vector>

class DQMStore;
class SiStripDetCabling;
class SiStripCluster;
class SiStripDCSStatus;
class GenericTriggerEventFlag;

class OuterTrackerClusters : public edm::EDAnalyzer {

public:
  explicit OuterTrackerClusters(const edm::ParameterSet&);
  ~OuterTrackerClusters();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
	// TTCluster stacks (pure cluster!)
	MonitorElement* Cluster_IMem_Barrel = 0;
	MonitorElement* Cluster_IMem_Endcap = 0;
	MonitorElement* Cluster_OMem_Barrel = 0;
	MonitorElement* Cluster_OMem_Endcap = 0;
	MonitorElement* Cluster_Gen_Barrel = 0;
	MonitorElement* Cluster_Unkn_Barrel = 0;
	MonitorElement* Cluster_Comb_Barrel = 0;
	MonitorElement* Cluster_Gen_Endcap = 0;
	MonitorElement* Cluster_Unkn_Endcap = 0;
	MonitorElement* Cluster_Comb_Endcap = 0;
	

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::InputTag clusterProducerStrip_;

  std::string topFolderName_;
  
};
#endif
