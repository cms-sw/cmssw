#ifndef OuterTrackerV_OuterTrackerV_h
#define OuterTrackerV_OuterTrackerV_h

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

class OuterTrackerV : public edm::EDAnalyzer {

public:
  explicit OuterTrackerV(const edm::ParameterSet&);
  ~OuterTrackerV();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
  
	// TrackingParticle and TrackingVertex
	MonitorElement* TPart_Pt = 0;
	MonitorElement* TPart_Eta_Pt10 = 0;
	MonitorElement* TPart_Phi_Pt10 = 0;
	MonitorElement* SimVtx_XY = 0;
	MonitorElement* SimVtx_RZ = 0;
	// CW vs. TPart Eta
	MonitorElement* TPart_Eta_INormalization = 0;
	MonitorElement* TPart_Eta_ICW_1 = 0;
	MonitorElement* TPart_Eta_ICW_2 = 0;
	MonitorElement* TPart_Eta_ICW_3 = 0;
	MonitorElement* TPart_Eta_ONormalization = 0;
	MonitorElement* TPart_Eta_OCW_1 = 0;
	MonitorElement* TPart_Eta_OCW_2 = 0;
	MonitorElement* TPart_Eta_OCW_3 = 0;
	MonitorElement* TPart_AbsEta_INormalization = 0;
	MonitorElement* TPart_AbsEta_ICW_1 = 0;
	MonitorElement* TPart_AbsEta_ICW_2 = 0;
	MonitorElement* TPart_AbsEta_ICW_3 = 0;
	MonitorElement* TPart_AbsEta_ONormalization = 0;
	MonitorElement* TPart_AbsEta_OCW_1 = 0;
	MonitorElement* TPart_AbsEta_OCW_2 = 0;
	MonitorElement* TPart_AbsEta_OCW_3 = 0;
	// Stub in PS/2S module vs. TPart Eta
	MonitorElement* TPart_Eta_Pt10_Normalization = 0;
	MonitorElement* TPart_Eta_Pt10_NumPS = 0;
	MonitorElement* TPart_Eta_Pt10_Num2S = 0;
	MonitorElement* TPart_AbsEta_Pt10_Normalization = 0;
	MonitorElement* TPart_AbsEta_Pt10_NumPS = 0;
	MonitorElement* TPart_AbsEta_Pt10_Num2S = 0;
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
