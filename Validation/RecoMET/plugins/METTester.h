#ifndef METTESTER_H
#define METTESTER_H

// author: Mike Schmitt (The University of Florida)
// date: 8/24/2006
// modification: Bobby Scurlock
// date: 03.11.2006
// note: added RMS(METx) vs SumET capability
// modification: Rick Cavanaugh
// date: 05.11.2006
// note: added configuration parameters
// modification: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

// Rewritten by Viola Sordini, Matthias Artur Weber, Robert Schoefbeck Nov./Dez.
// 2013

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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class METTester : public DQMEDAnalyzer {
public:
  explicit METTester(const edm::ParameterSet &);

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  static const int mNMETBins = 11;
  static constexpr std::array<float, mNMETBins+1> mMETBins = {{0., 20., 40., 60., 80., 100., 150., 200., 300., 400., 500., 10000.}};
  static const int mNEtaBins = 39;
  static constexpr std::array<float, mNEtaBins+1> mEtaBins = {{-6., -5.5, -4., -4.5, -4, -3.5, -3., -2.75, -2.5, -2.25, -2., -1.75, -1.5, -1.25, -1., -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 3.5, 4., 4.5, 5., 5.5, 6.}};
  static const int mNPhiBins = 14;
  static constexpr std::array<float, mNPhiBins+1> mPhiBins = {{-3.15, -3., -2.5, -2., -1.5, -1., -0.5, -0., 0.5, 1., 1.5, 2., 2.5, 3., 3.15}};

private:
  std::map<std::string, MonitorElement *> me;

  // Inputs from Configuration File
  edm::InputTag mInputCollection_;
  edm::InputTag inputMETLabel_;
  std::string METType_;
  edm::InputTag inputCaloMETLabel_;

  // Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> caloMETsToken_;
  edm::EDGetTokenT<reco::PFMETCollection> pfMETsToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genMETsToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genMETsTrueToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genMETsCaloToken_;
  edm::EDGetTokenT<pat::METCollection> patMETToken_;

  // Events variables
  MonitorElement *mNvertex;

  // Common variables
  MonitorElement *mMEx;
  MonitorElement *mMEy;
  MonitorElement *mMETPseudoSign;
  MonitorElement *mMETRealSign;
  MonitorElement *mMET;
  MonitorElement *mMETFine;
  MonitorElement *mMET_Nvtx;
  MonitorElement *mMETEta;
  MonitorElement *mMETPhi;
  MonitorElement *mSumET;
  MonitorElement *mMETDiff_GenMETTrue;
  MonitorElement *mMETRatio_GenMETTrue;
  MonitorElement *mMETDeltaPhi_GenMETTrue;
  MonitorElement *mMETDiff_GenMETCalo;
  MonitorElement *mMETRatio_GenMETCalo;
  MonitorElement *mMETDeltaPhi_GenMETCalo;

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
  MonitorElement *mCaloSETInpHF;
  MonitorElement *mCaloSETInmHF;
  MonitorElement *mCaloEmEtInEE;
  MonitorElement *mCaloEmEtInEB;

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
  
  ElemArr<mNMETBins> mMET_METBins;
  ElemArr<mNEtaBins> mMET_EtaBins;
  ElemArr<mNPhiBins> mMET_PhiBins;
  
  ElemArr<mNMETBins> mMETDiff_GenMETTrue_METBins;
  ElemArr<mNEtaBins> mMETDiff_GenMETTrue_EtaBins;
  ElemArr<mNPhiBins> mMETDiff_GenMETTrue_PhiBins;
  ElemArr<mNMETBins> mMETRatio_GenMETTrue_METBins;
  ElemArr<mNEtaBins> mMETRatio_GenMETTrue_EtaBins;
  ElemArr<mNPhiBins> mMETRatio_GenMETTrue_PhiBins;
  ElemArr<mNMETBins> mMETDeltaPhi_GenMETTrue_METBins;
  ElemArr<mNEtaBins> mMETDeltaPhi_GenMETTrue_EtaBins;
  ElemArr<mNPhiBins> mMETDeltaPhi_GenMETTrue_PhiBins;

  bool isCaloMET;
  bool isPFMET;
  bool isGenMET;
  bool isMiniAODMET;
};

#endif  // METTESTER_H
