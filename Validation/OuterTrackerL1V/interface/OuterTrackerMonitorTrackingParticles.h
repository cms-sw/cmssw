#ifndef OuterTrackerL1V_OuterTrackerMonitorTrackingParticles_h
#define OuterTrackerL1V_OuterTrackerMonitorTrackingParticles_h

#include <vector>
#include <memory>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "DataFormats/Common/interface/DetSetVector.h"


class DQMStore;

class OuterTrackerMonitorTrackingParticles : public DQMEDAnalyzer {

public:
  explicit OuterTrackerMonitorTrackingParticles(const edm::ParameterSet&);
  ~OuterTrackerMonitorTrackingParticles() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // Tracking particle distributions
  MonitorElement* trackParts_Eta = 0;
  MonitorElement* trackParts_Phi = 0;
  MonitorElement* trackParts_Pt = 0;

  // Plots for correctly matched tracks
  MonitorElement* Track_MatchedChi2 = 0; //Chi2 for only tracks correctly matched to truth level
  MonitorElement* Track_MatchedChi2Red = 0; //Chi2/dof for only tracks correctly matched to truth level

  // pT and eta for efficiency plots
  MonitorElement* tp_pt = 0;  // denominator
  MonitorElement* tp_eta = 0; // denominator
  MonitorElement* tp_d0 = 0; // denominator
  MonitorElement* tp_VtxR = 0; // denominator (also known as vxy)
  MonitorElement* tp_VtxZ = 0; // denominator
  MonitorElement* match_tp_pt = 0; // numerator
  MonitorElement* match_tp_eta = 0; // numerator
  MonitorElement* match_tp_d0 = 0; // numerator
  MonitorElement* match_tp_VtxR = 0; // numerator (also known as vxy)
  MonitorElement* match_tp_VtxZ = 0; // numerator

  // 1D intermediate resolution plots (pT and eta)
  MonitorElement* res_eta = 0; // for all eta and pT
  MonitorElement* res_pt = 0; // for all eta and pT
  MonitorElement* res_ptRel = 0; // for all eta and pT (delta(pT)/pT)
  MonitorElement* respt_eta0to0p7_pt2to3 = 0;
  MonitorElement* respt_eta0p7to1_pt2to3 = 0;
  MonitorElement* respt_eta1to1p2_pt2to3 = 0;
  MonitorElement* respt_eta1p2to1p6_pt2to3 = 0;
  MonitorElement* respt_eta1p6to2_pt2to3 = 0;
  MonitorElement* respt_eta2to2p4_pt2to3 = 0;
  MonitorElement* respt_eta0to0p7_pt3to8 = 0;
  MonitorElement* respt_eta0p7to1_pt3to8 = 0;
  MonitorElement* respt_eta1to1p2_pt3to8 = 0;
  MonitorElement* respt_eta1p2to1p6_pt3to8 = 0;
  MonitorElement* respt_eta1p6to2_pt3to8 = 0;
  MonitorElement* respt_eta2to2p4_pt3to8 = 0;
  MonitorElement* respt_eta0to0p7_pt8toInf = 0;
  MonitorElement* respt_eta0p7to1_pt8toInf = 0;
  MonitorElement* respt_eta1to1p2_pt8toInf = 0;
  MonitorElement* respt_eta1p2to1p6_pt8toInf = 0;
  MonitorElement* respt_eta1p6to2_pt8toInf = 0;
  MonitorElement* respt_eta2to2p4_pt8toInf = 0;
  MonitorElement* reseta_eta0to0p7 = 0;
  MonitorElement* reseta_eta0p7to1 = 0;
  MonitorElement* reseta_eta1to1p2 = 0;
  MonitorElement* reseta_eta1p2to1p6 = 0;
  MonitorElement* reseta_eta1p6to2 = 0;
  MonitorElement* reseta_eta2to2p4 = 0;
  MonitorElement* resphi_eta0to0p7 = 0;
  MonitorElement* resphi_eta0p7to1 = 0;
  MonitorElement* resphi_eta1to1p2 = 0;
  MonitorElement* resphi_eta1p2to1p6 = 0;
  MonitorElement* resphi_eta1p6to2 = 0;
  MonitorElement* resphi_eta2to2p4 = 0;
  MonitorElement* resVtxZ_eta0to0p7 = 0;
  MonitorElement* resVtxZ_eta0p7to1 = 0;
  MonitorElement* resVtxZ_eta1to1p2 = 0;
  MonitorElement* resVtxZ_eta1p2to1p6 = 0;
  MonitorElement* resVtxZ_eta1p6to2 = 0;
  MonitorElement* resVtxZ_eta2to2p4 = 0;

  // For d0
  MonitorElement* resd0_eta0to0p7 = 0;
  MonitorElement* resd0_eta0p7to1 = 0;
  MonitorElement* resd0_eta1to1p2 = 0;
  MonitorElement* resd0_eta1p2to1p6 = 0;
  MonitorElement* resd0_eta1p6to2 = 0;
  MonitorElement* resd0_eta2to2p4 = 0;

 private:
   edm::ParameterSet conf_;
   edm::EDGetTokenT< std::vector <TrackingParticle> > trackingParticleToken_;
   edm::EDGetTokenT< edmNew::DetSetVector < TTStub < Ref_Phase2TrackerDigi_ > > > ttStubToken_;
   edm::EDGetTokenT<std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >  ttTrackToken_;
   edm::EDGetTokenT< TTClusterAssociationMap< Ref_Phase2TrackerDigi_ > > ttClusterMCTruthToken_;//MC truth association map for clusters
   edm::EDGetTokenT< TTStubAssociationMap < Ref_Phase2TrackerDigi_ > > ttStubMCTruthToken_; //MC truth association map for stubs
   edm::EDGetTokenT< TTTrackAssociationMap< Ref_Phase2TrackerDigi_ > > ttTrackMCTruthToken_;//MC truth association map for tracks
   int L1Tk_nPar;
   int L1Tk_minNStub;
   double L1Tk_maxChi2;
   double L1Tk_maxChi2dof;
   int TP_minNStub;
   double TP_minPt;
   double TP_maxPt;
   double TP_maxEta;
   double TP_maxVtxZ;
   int TP_select_eventid;
   std::string topFolderName_;
};
#endif
