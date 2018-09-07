#ifndef SiOuterTrackerV_OuterTrackerMonitorTrackingParticles_h
#define SiOuterTrackerV_OuterTrackerMonitorTrackingParticles_h

#include <vector>
#include <memory>
#include <string>
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
  int Layer(const float R_, const float Z_) const;


  // Tracking particle distributions
  MonitorElement* trackParts_Eta = nullptr;
  MonitorElement* trackParts_Phi = nullptr;
  MonitorElement* trackParts_Pt = nullptr;

  // Plots for correctly matched tracks
  MonitorElement* Track_MatchedChi2 = nullptr; //Chi2 for only tracks correctly matched to truth level
  MonitorElement* Track_MatchedChi2Red = nullptr; //Chi2/dof for only tracks correctly matched to truth level

  // pT and eta for efficiency plots
  MonitorElement* tp_pt = nullptr;  // denominator
  MonitorElement* tp_pt_zoom = nullptr;  // denominator
  MonitorElement* tp_eta = nullptr; // denominator
  MonitorElement* tp_d0 = nullptr; // denominator
  MonitorElement* tp_VtxR = nullptr; // denominator (also known as vxy)
  MonitorElement* tp_VtxZ = nullptr; // denominator
  MonitorElement* match_tp_pt = nullptr; // numerator
  MonitorElement* match_tp_pt_zoom = nullptr; // numerator
  MonitorElement* match_tp_eta = nullptr; // numerator
  MonitorElement* match_tp_d0 = nullptr; // numerator
  MonitorElement* match_tp_VtxR = nullptr; // numerator (also known as vxy)
  MonitorElement* match_tp_VtxZ = nullptr; // numerator

  // 1D intermediate resolution plots (pT and eta)
  MonitorElement* res_eta = nullptr; // for all eta and pT
  MonitorElement* res_pt = nullptr; // for all eta and pT
  MonitorElement* res_ptRel = nullptr; // for all eta and pT (delta(pT)/pT)
  MonitorElement* respt_eta0to0p7_pt2to3 = nullptr;
  MonitorElement* respt_eta0p7to1_pt2to3 = nullptr;
  MonitorElement* respt_eta1to1p2_pt2to3 = nullptr;
  MonitorElement* respt_eta1p2to1p6_pt2to3 = nullptr;
  MonitorElement* respt_eta1p6to2_pt2to3 = nullptr;
  MonitorElement* respt_eta2to2p4_pt2to3 = nullptr;
  MonitorElement* respt_eta0to0p7_pt3to8 = nullptr;
  MonitorElement* respt_eta0p7to1_pt3to8 = nullptr;
  MonitorElement* respt_eta1to1p2_pt3to8 = nullptr;
  MonitorElement* respt_eta1p2to1p6_pt3to8 = nullptr;
  MonitorElement* respt_eta1p6to2_pt3to8 = nullptr;
  MonitorElement* respt_eta2to2p4_pt3to8 = nullptr;
  MonitorElement* respt_eta0to0p7_pt8toInf = nullptr;
  MonitorElement* respt_eta0p7to1_pt8toInf = nullptr;
  MonitorElement* respt_eta1to1p2_pt8toInf = nullptr;
  MonitorElement* respt_eta1p2to1p6_pt8toInf = nullptr;
  MonitorElement* respt_eta1p6to2_pt8toInf = nullptr;
  MonitorElement* respt_eta2to2p4_pt8toInf = nullptr;
  MonitorElement* reseta_eta0to0p7 = nullptr;
  MonitorElement* reseta_eta0p7to1 = nullptr;
  MonitorElement* reseta_eta1to1p2 = nullptr;
  MonitorElement* reseta_eta1p2to1p6 = nullptr;
  MonitorElement* reseta_eta1p6to2 = nullptr;
  MonitorElement* reseta_eta2to2p4 = nullptr;
  MonitorElement* resphi_eta0to0p7 = nullptr;
  MonitorElement* resphi_eta0p7to1 = nullptr;
  MonitorElement* resphi_eta1to1p2 = nullptr;
  MonitorElement* resphi_eta1p2to1p6 = nullptr;
  MonitorElement* resphi_eta1p6to2 = nullptr;
  MonitorElement* resphi_eta2to2p4 = nullptr;
  MonitorElement* resVtxZ_eta0to0p7 = nullptr;
  MonitorElement* resVtxZ_eta0p7to1 = nullptr;
  MonitorElement* resVtxZ_eta1to1p2 = nullptr;
  MonitorElement* resVtxZ_eta1p2to1p6 = nullptr;
  MonitorElement* resVtxZ_eta1p6to2 = nullptr;
  MonitorElement* resVtxZ_eta2to2p4 = nullptr;

  // For d0
  MonitorElement* resd0_eta0to0p7 = nullptr;
  MonitorElement* resd0_eta0p7to1 = nullptr;
  MonitorElement* resd0_eta1to1p2 = nullptr;
  MonitorElement* resd0_eta1p2to1p6 = nullptr;
  MonitorElement* resd0_eta1p6to2 = nullptr;
  MonitorElement* resd0_eta2to2p4 = nullptr;

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

   // The following adds a variable that is the number of layers hit for each tracking particle
   struct TpStruct{
     int TpId;
     std::vector<bool> layer;
     int Nlayers(){ //Counts how many layers are set to "true" for their hit status
       int layers=0;
       for(unsigned l=0; l<layer.size(); ++l) if(layer[l]) ++layers;
       return layers;
     }
   };

};
#endif
