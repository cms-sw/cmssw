#ifndef METTESTER_H
#define METTESTER_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TMath.h"
#include "TVector2.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class METTester : public DQMEDAnalyzer {
public:
  explicit METTester(const edm::ParameterSet &);

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  std::map<std::string, MonitorElement *> me;

  // Inputs from Configuration File
  edm::InputTag mInputCollection_;
  std::string inputMETLabel_;
  std::string inputMHTLabel_;
  std::string METType_;
  edm::InputTag inputCaloMETLabel_;

  // Tokens
  edm::InputTag pvTokenTag_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> caloMetToken_;
  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken_;
  edm::EDGetTokenT<reco::METCollection> recoMhtToken_;
  edm::EDGetTokenT<reco::METCollection> genMhtToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genMetToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genMetTrueToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genMetCaloToken_;
  edm::EDGetTokenT<pat::METCollection> patMetToken_;

  edm::Handle<reco::CaloMETCollection> caloMetHandle_;
  edm::Handle<reco::PFMETCollection> pfMetHandle_;
  edm::Handle<reco::GenMETCollection> genMetHandle_;
  edm::Handle<pat::METCollection> patMetHandle_;
  edm::Handle<reco::METCollection> genMhtHandle_;
  edm::Handle<reco::METCollection> recoMhtHandle_;

  // Events variables
  MonitorElement *mNvertex;

  // Common variables
  MonitorElement *mMEx;
  MonitorElement *mMEy;
  MonitorElement *mMETSignPseudo;
  MonitorElement *mMETSignReal;
  MonitorElement *mGenMETTrue;
  MonitorElement *mGenMETCalo;
  MonitorElement *mMET;
  MonitorElement *mMHT;
  MonitorElement *mMET_Nvtx;
  MonitorElement *mMETPhi;
  MonitorElement *mSumET;
  MonitorElement *mMETDiff_GenMETTrue;
  MonitorElement *mMETRatio_GenMETTrue;
  MonitorElement *mMETDeltaPhi_GenMETTrue;
  MonitorElement *mMETDiff_GenMETCalo;
  MonitorElement *mMETRatio_GenMETCalo;
  MonitorElement *mMETDeltaPhi_GenMETCalo;

  MonitorElement *mMET_vs_MHT;
  MonitorElement *mGenMETTrue_vs_GenMHT;

  MonitorElement *mGenMETTrue_vs_MET;
  MonitorElement *mGenMETPhi_vs_MET;
  MonitorElement *mGenMETTrue_vs_mGenMETPhi;
  MonitorElement *mMETDiff_vs_GenMETTrue;
  MonitorElement *mMETDiff_vs_GenMETPhi;
  MonitorElement *mMETRatio_vs_GenMETTrue;
  MonitorElement *mMETRatio_vs_GenMETPhi;  
  MonitorElement *mMETDeltaPhi_vs_GenMETTrue;
  MonitorElement *mMETDeltaPhi_vs_GenMETPhi;  

  // MET Uncertainity Variables
  MonitorElement *mMETUnc_JetResUp;
  MonitorElement *mMETUnc_JetResDown;
  MonitorElement *mMETUnc_JetEnUp;
  MonitorElement *mMETUnc_JetEnDown;
  MonitorElement *mMETUnc_MuonEnUp;
  MonitorElement *mMETUnc_MuonEnDown;
  MonitorElement *mMETUnc_ElectronEnUp;
  MonitorElement *mMETUnc_ElectronEnDown;
  MonitorElement *mMETUnc_TauEnUp;
  MonitorElement *mMETUnc_TauEnDown;
  MonitorElement *mMETUnc_UnclusteredEnUp;
  MonitorElement *mMETUnc_UnclusteredEnDown;
  MonitorElement *mMETUnc_PhotonEnUp;
  MonitorElement *mMETUnc_PhotonEnDown;

  // CaloMET variables
  MonitorElement *mCaloMaxEtInEmTowers;
  MonitorElement *mCaloMaxEtInHadTowers;
  MonitorElement *mCaloEtFractionHadronic;
  MonitorElement *mCaloEmEtFraction;
  MonitorElement *mCaloHadEtInHB;
  MonitorElement *mCaloHadEtInHO;
  MonitorElement *mCaloHadEtInHE;
  MonitorElement *mCaloHadEtInHF;
  MonitorElement *mCaloHadEtInEB;
  MonitorElement *mCaloHadEtInEE;
  MonitorElement *mCaloEmEtInHF;
  MonitorElement *mCaloEmEtInEE;
  MonitorElement *mCaloEmEtInEB;

  MonitorElement *mGenMETCalo_vs_MET;
  MonitorElement *mGenMETCalo_vs_mGenMETPhi;

  // GenMET variables
  MonitorElement *mNeutralEMEtFraction;
  MonitorElement *mNeutralHadEtFraction;
  MonitorElement *mChargedEMEtFraction;
  MonitorElement *mChargedHadEtFraction;
  MonitorElement *mMuonEtFraction;
  MonitorElement *mInvisibleEtFraction;

  // PFMET variables
  MonitorElement *mPFphotonEtFraction;
  MonitorElement *mPFphotonEt;
  MonitorElement *mPFneutralHadronEtFraction;
  MonitorElement *mPFneutralHadronEt;
  MonitorElement *mPFelectronEtFraction;
  MonitorElement *mPFelectronEt;
  MonitorElement *mPFchargedHadronEtFraction;
  MonitorElement *mPFchargedHadronEt;
  MonitorElement *mPFmuonEtFraction;
  MonitorElement *mPFmuonEt;
  MonitorElement *mPFHFHadronEtFraction;
  MonitorElement *mPFHFHadronEt;
  MonitorElement *mPFHFEMEtFraction;
  MonitorElement *mPFHFEMEt;

  template <size_t S>
  using ElemArr = std::array<MonitorElement *, S>;

  bool isCaloMET;
  bool isPFMET;
  bool isMHT;
  bool isGenMET;
  bool isMiniAODMET;
  std::string runDir;
  std::string mGenMetTrueLabel;
  std::string mGenMetCaloLabel;

  float phiEdge = 3.2;
  float metEdge = 1600.;
  float metDiffEdge = 600.;
  float metRatioEdge = 600.;

  int phiNbins = 100;
  int metNbins = 1600;
  int metDiffNbins = 600;
  int metRatioNbins = 600;
};

#endif  // METTESTER_H
