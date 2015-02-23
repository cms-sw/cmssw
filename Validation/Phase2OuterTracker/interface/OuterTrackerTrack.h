#ifndef Phase2OuterTracker_OuterTrackerTrack_h
#define Phase2OuterTracker_OuterTrackerTrack_h

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

class OuterTrackerTrack : public edm::EDAnalyzer {

public:
  explicit OuterTrackerTrack(const edm::ParameterSet&);
  ~OuterTrackerTrack();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  //virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  
  
	MonitorElement* Track_NStubs = 0;
  /// TTTracks made from up to 2 TTStubs
	MonitorElement* Track_2Stubs_N = 0;
	MonitorElement* Track_2Stubs_Pt = 0;
	MonitorElement* Track_2Stubs_Eta = 0;
	MonitorElement* Track_2Stubs_Phi = 0;
  MonitorElement* Track_2Stubs_VtxZ0 = 0;
  MonitorElement* Track_2Stubs_Chi2 = 0;
  MonitorElement* Track_2Stubs_Chi2Red = 0;
  MonitorElement* Track_2Stubs_Chi2_NStubs = 0;
	//MonitorElement* Track_2Stubs_Chi2_TPart_Eta = 0;
	MonitorElement* Track_2Stubs_Chi2Red_NStubs = 0;
	//MonitorElement* Track_2Stubs_Chi2Red_TPart_Eta = 0;
	/// Resolution plots
	MonitorElement* Track_2Stubs_Pt_TPart_Pt = 0;
	MonitorElement* Track_2Stubs_PtRes_TPart_Eta = 0;
	MonitorElement* Track_2Stubs_InvPt_TPart_InvPt = 0;
	MonitorElement* Track_2Stubs_InvPtRes_TPart_Eta = 0;
	MonitorElement* Track_2Stubs_Phi_TPart_Phi = 0;
	MonitorElement* Track_2Stubs_PhiRes_TPart_Eta = 0;
	MonitorElement* Track_2Stubs_Eta_TPart_Eta = 0;
	MonitorElement* Track_2Stubs_EtaRes_TPart_Eta = 0;
	MonitorElement* Track_2Stubs_VtxZ0_TPart_VtxZ0 = 0;
	MonitorElement* Track_2Stubs_VtxZ0Res_TPart_Eta = 0;
  
  /// TTTracks made from at least 3 TTStubs
	MonitorElement* Track_3Stubs_N = 0;
	MonitorElement* Track_3Stubs_Pt = 0;
	MonitorElement* Track_3Stubs_Eta = 0;
	MonitorElement* Track_3Stubs_Phi = 0;
  MonitorElement* Track_3Stubs_VtxZ0 = 0;
  MonitorElement* Track_3Stubs_Chi2 = 0;
  MonitorElement* Track_3Stubs_Chi2Red = 0;
  MonitorElement* Track_3Stubs_Chi2_NStubs = 0;
	//MonitorElement* Track_3Stubs_Chi2_TPart_Eta = 0;
	MonitorElement* Track_3Stubs_Chi2Red_NStubs = 0;
	//MonitorElement* Track_3Stubs_Chi2Red_TPart_Eta = 0;
	///Resolution plots
	MonitorElement* Track_3Stubs_Pt_TPart_Pt = 0;
	MonitorElement* Track_3Stubs_PtRes_TPart_Eta = 0;
	MonitorElement* Track_3Stubs_InvPt_TPart_InvPt = 0;
	MonitorElement* Track_3Stubs_InvPtRes_TPart_Eta = 0;
	MonitorElement* Track_3Stubs_Phi_TPart_Phi = 0;
	MonitorElement* Track_3Stubs_PhiRes_TPart_Eta = 0;
	MonitorElement* Track_3Stubs_Eta_TPart_Eta = 0;
	MonitorElement* Track_3Stubs_EtaRes_TPart_Eta = 0;
	MonitorElement* Track_3Stubs_VtxZ0_TPart_VtxZ0 = 0;
	MonitorElement* Track_3Stubs_VtxZ0Res_TPart_Eta = 0;
  
  
 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
  
};
#endif
