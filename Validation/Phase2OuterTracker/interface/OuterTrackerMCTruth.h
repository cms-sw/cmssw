#ifndef Phase2OuterTracker_OuterTrackerMCTruth_h
#define Phase2OuterTracker_OuterTrackerMCTruth_h

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

class DQMStore;

class OuterTrackerMCTruth : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMCTruth(const edm::ParameterSet&);
  ~OuterTrackerMCTruth();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
  
  // TrackingParticle and TrackingVertex
  MonitorElement* SimVtx_XY = 0;
  MonitorElement* SimVtx_RZ = 0;
  
  MonitorElement* TPart_Pt = 0;
  MonitorElement* TPart_Eta_Pt10 = 0;
  MonitorElement* TPart_Phi_Pt10 = 0;
  
  MonitorElement* TPart_Cluster_Pt = 0;
  MonitorElement* TPart_Cluster_Phi_Pt10 = 0;
  MonitorElement* TPart_Cluster_Eta_Pt10 = 0;
  
  MonitorElement* TPart_Stub_Pt = 0;
  MonitorElement* TPart_Stub_Phi_Pt10 = 0;
  MonitorElement* TPart_Stub_Eta_Pt10 = 0;
  
  MonitorElement* TPart_Track_LQ_Pt = 0;
  MonitorElement* TPart_Track_LQ_Phi_Pt10 = 0;
  MonitorElement* TPart_Track_LQ_Eta_Pt10 = 0;
  MonitorElement* TPart_Track_HQ_Pt = 0;
  MonitorElement* TPart_Track_HQ_Phi_Pt10 = 0;
  MonitorElement* TPart_Track_HQ_Eta_Pt10 = 0;
  
  // Stub in PS/2S module vs. TPart Eta
  MonitorElement* TPart_Stub_Eta_Pt10_Normalization = 0;
  MonitorElement* TPart_Stub_Eta_Pt10_NumPS = 0;
  MonitorElement* TPart_Stub_Eta_Pt10_Num2S = 0;
  
  // CW vs. TPart Eta
  MonitorElement* TPart_Eta_INormalization = 0;
  MonitorElement* TPart_Eta_ICW_1 = 0;
  MonitorElement* TPart_Eta_ICW_2 = 0;
  MonitorElement* TPart_Eta_ICW_3 = 0;
  MonitorElement* TPart_Eta_ONormalization = 0;
  MonitorElement* TPart_Eta_OCW_1 = 0;
  MonitorElement* TPart_Eta_OCW_2 = 0;
  MonitorElement* TPart_Eta_OCW_3 = 0;
  
  // PID
  MonitorElement* Cluster_PID = 0;
  MonitorElement* Stub_PID = 0;
  
  // Track Chi2(Red)
  MonitorElement* Track_LQ_Chi2_TPart_Eta = 0;
  MonitorElement* Track_LQ_Chi2Red_TPart_Eta = 0;
  MonitorElement* Track_HQ_Chi2_TPart_Eta = 0;
  MonitorElement* Track_HQ_Chi2Red_TPart_Eta = 0;
  
  
  // Stubs vs. TrackingParticles
  MonitorElement* Stub_InvPt_TPart_InvPt_AllLayers = 0;
  MonitorElement* Stub_Pt_TPart_Pt_AllLayers = 0;
  MonitorElement* Stub_Eta_TPart_Eta_AllLayers = 0;
  MonitorElement* Stub_Phi_TPart_Phi_AllLayers = 0;
  
  MonitorElement* Stub_InvPtRes_TPart_Eta_AllLayers = 0;
  MonitorElement* Stub_PtRes_TPart_Eta_AllLayers = 0;
  MonitorElement* Stub_EtaRes_TPart_Eta_AllLayers = 0;
  MonitorElement* Stub_PhiRes_TPart_Eta_AllLayers = 0;
  
  MonitorElement* Stub_W_TPart_Pt_AllLayers = 0;
  MonitorElement* Stub_W_TPart_InvPt_AllLayers = 0;
  
  MonitorElement* Stub_InvPt_TPart_InvPt_AllDisks = 0;
  MonitorElement* Stub_Pt_TPart_Pt_AllDisks = 0;
  MonitorElement* Stub_Eta_TPart_Eta_AllDisks = 0;
  MonitorElement* Stub_Phi_TPart_Phi_AllDisks = 0;
  
  MonitorElement* Stub_InvPtRes_TPart_Eta_AllDisks = 0;
  MonitorElement* Stub_PtRes_TPart_Eta_AllDisks = 0;
  MonitorElement* Stub_EtaRes_TPart_Eta_AllDisks = 0;
  MonitorElement* Stub_PhiRes_TPart_Eta_AllDisks = 0;
  
  MonitorElement* Stub_W_TPart_Pt_AllDisks = 0;
  MonitorElement* Stub_W_TPart_InvPt_AllDisks = 0;
  
  
  // Tracks vs. TrackingParticles
  MonitorElement* Track_LQ_Pt_TPart_Pt = 0;
  MonitorElement* Track_LQ_PtRes_TPart_Eta = 0;
  MonitorElement* Track_LQ_InvPt_TPart_InvPt = 0;
  MonitorElement* Track_LQ_InvPtRes_TPart_Eta = 0;
  MonitorElement* Track_LQ_Phi_TPart_Phi = 0;
  MonitorElement* Track_LQ_PhiRes_TPart_Eta = 0;
  MonitorElement* Track_LQ_Eta_TPart_Eta = 0;
  MonitorElement* Track_LQ_EtaRes_TPart_Eta = 0;
  MonitorElement* Track_LQ_VtxZ0_TPart_VtxZ0 = 0;
  MonitorElement* Track_LQ_VtxZ0Res_TPart_Eta = 0;
  
  MonitorElement* Track_HQ_Pt_TPart_Pt = 0;
  MonitorElement* Track_HQ_PtRes_TPart_Eta = 0;
  MonitorElement* Track_HQ_InvPt_TPart_InvPt = 0;
  MonitorElement* Track_HQ_InvPtRes_TPart_Eta = 0;
  MonitorElement* Track_HQ_Phi_TPart_Phi = 0;
  MonitorElement* Track_HQ_PhiRes_TPart_Eta = 0;
  MonitorElement* Track_HQ_Eta_TPart_Eta = 0;
  MonitorElement* Track_HQ_EtaRes_TPart_Eta = 0;
  MonitorElement* Track_HQ_VtxZ0_TPart_VtxZ0 = 0;
  MonitorElement* Track_HQ_VtxZ0Res_TPart_Eta = 0;
  
  
 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::InputTag tagTTClusters_;
  edm::InputTag tagTTClusterMCTruth_;
  edm::InputTag tagTTStubs_;
  edm::InputTag tagTTStubMCTruth_;
  edm::InputTag tagTTTracks_;
  edm::InputTag tagTTTrackMCTruth_;

  std::string topFolderName_;
  unsigned int HQDelim_;
  bool verbosePlots_;
  
};
#endif
