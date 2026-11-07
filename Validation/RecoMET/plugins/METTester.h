#ifndef METTESTER_H
#define METTESTER_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "TVector2.h"

#include <set>
#include <string>

// Validation of a single MET-like object (MET or MHT), compared to its generator-level reference.
// The object being validated is the "primary" collection: `inputMETLabel` for MET types, `inputMHTLabel` for "mht".
// Histograms are booked in `<runDir>/<outFolder>`, or `<runDir>/<module label>` if `outFolder` is empty.
// The other label, if not empty, defines a "companion" collection used to fill MET vs MHT plots
// in a dedicated `<runDir>/METvsMHT/<MET label>_<MHT label>` folder.
class METTester : public DQMEDAnalyzer {
public:
  explicit METTester(const edm::ParameterSet &);

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  using METView = edm::View<reco::MET>;

  // returns the first element of the collection, or nullptr (and warns once) if it is missing or empty
  const reco::MET *firstOrNull(const edm::Event &, const edm::EDGetTokenT<METView> &, const std::string &label);
  void warnOnce(const std::string &label, const std::string &msg);

  void fillGenComparison(const reco::MET &reco, const reco::MET &gen);
  void fillGenCaloComparison(const reco::MET &reco, const reco::MET &genCalo);
  void fillCaloMET(const reco::CaloMET &);
  void fillPFMET(const reco::PFMET &);
  void fillGenMET(const reco::GenMET &);
  void fillPatMET(const pat::MET &, double met);

  // Inputs from Configuration File
  std::string inputMETLabel_;
  std::string inputMHTLabel_;
  std::string METType_;
  std::string runDir;
  std::string outFolder_;
  std::string mGenMetTrueLabel;
  std::string mGenMetCaloLabel;
  std::string mGenMhtLabel;
  edm::InputTag pvTokenTag_;

  bool isCaloMET;
  bool isPFMET;
  bool isMHT;
  bool isGenMET;
  bool isMiniAODMET;

  // primary: the object being validated; companion: optional MHT (MET) partner of a MET (MHT)
  std::string primaryLabel_;
  std::string companionLabel_;
  bool hasCompanion_;

  // Tokens
  edm::EDGetTokenT<std::vector<reco::Vertex>> pvToken_;
  edm::EDGetTokenT<METView> primaryToken_;
  edm::EDGetTokenT<METView> companionToken_;
  edm::EDGetTokenT<METView> genMetTrueToken_;
  edm::EDGetTokenT<METView> genMetCaloToken_;
  edm::EDGetTokenT<METView> genMhtToken_;

  // labels of missing inputs already reported, to avoid flooding the logs
  std::set<std::string> warned_;

  // Events variables
  MonitorElement *mNvertex = nullptr;

  // Common variables
  MonitorElement *mMEx = nullptr;
  MonitorElement *mMEy = nullptr;
  MonitorElement *mMETSignPseudo = nullptr;
  MonitorElement *mMETSignReal = nullptr;
  MonitorElement *mGenMETTrue = nullptr;
  MonitorElement *mGenMETCalo = nullptr;
  MonitorElement *mMET = nullptr;
  MonitorElement *mMET_Nvtx = nullptr;
  MonitorElement *mMETPhi = nullptr;
  MonitorElement *mSumET = nullptr;
  MonitorElement *mMETDiff_GenMETTrue = nullptr;
  MonitorElement *mMETRatio_GenMETTrue = nullptr;
  MonitorElement *mMETDeltaPhi_GenMETTrue = nullptr;
  MonitorElement *mMETDiff_GenMETCalo = nullptr;
  MonitorElement *mMETRatio_GenMETCalo = nullptr;
  MonitorElement *mMETDeltaPhi_GenMETCalo = nullptr;

  // MET vs MHT variables (separate folder)
  MonitorElement *mMET_vs_MHT = nullptr;
  MonitorElement *mGenMETTrue_vs_GenMHT = nullptr;

  MonitorElement *mGenMETTrue_vs_MET = nullptr;
  MonitorElement *mGenMETPhi_vs_MET = nullptr;
  MonitorElement *mGenMETTrue_vs_mGenMETPhi = nullptr;
  MonitorElement *mMETDiff_vs_GenMETTrue = nullptr;
  MonitorElement *mMETDiff_vs_GenMETPhi = nullptr;
  MonitorElement *mMETRatio_vs_GenMETTrue = nullptr;
  MonitorElement *mMETRatio_vs_GenMETPhi = nullptr;
  MonitorElement *mMETDeltaPhi_vs_GenMETTrue = nullptr;
  MonitorElement *mMETDeltaPhi_vs_GenMETPhi = nullptr;

  // MET Uncertainity Variables
  MonitorElement *mMETUnc_JetResUp = nullptr;
  MonitorElement *mMETUnc_JetResDown = nullptr;
  MonitorElement *mMETUnc_JetEnUp = nullptr;
  MonitorElement *mMETUnc_JetEnDown = nullptr;
  MonitorElement *mMETUnc_MuonEnUp = nullptr;
  MonitorElement *mMETUnc_MuonEnDown = nullptr;
  MonitorElement *mMETUnc_ElectronEnUp = nullptr;
  MonitorElement *mMETUnc_ElectronEnDown = nullptr;
  MonitorElement *mMETUnc_TauEnUp = nullptr;
  MonitorElement *mMETUnc_TauEnDown = nullptr;
  MonitorElement *mMETUnc_UnclusteredEnUp = nullptr;
  MonitorElement *mMETUnc_UnclusteredEnDown = nullptr;
  MonitorElement *mMETUnc_PhotonEnUp = nullptr;
  MonitorElement *mMETUnc_PhotonEnDown = nullptr;

  // CaloMET variables
  MonitorElement *mCaloMaxEtInEmTowers = nullptr;
  MonitorElement *mCaloMaxEtInHadTowers = nullptr;
  MonitorElement *mCaloEtFractionHadronic = nullptr;
  MonitorElement *mCaloEmEtFraction = nullptr;
  MonitorElement *mCaloHadEtInHB = nullptr;
  MonitorElement *mCaloHadEtInHO = nullptr;
  MonitorElement *mCaloHadEtInHE = nullptr;
  MonitorElement *mCaloHadEtInHF = nullptr;
  MonitorElement *mCaloEmEtInHF = nullptr;
  MonitorElement *mCaloEmEtInEE = nullptr;
  MonitorElement *mCaloEmEtInEB = nullptr;

  MonitorElement *mGenMETCalo_vs_MET = nullptr;
  MonitorElement *mGenMETCalo_vs_mGenMETPhi = nullptr;

  // GenMET variables
  MonitorElement *mNeutralEMEtFraction = nullptr;
  MonitorElement *mNeutralHadEtFraction = nullptr;
  MonitorElement *mChargedEMEtFraction = nullptr;
  MonitorElement *mChargedHadEtFraction = nullptr;
  MonitorElement *mMuonEtFraction = nullptr;
  MonitorElement *mInvisibleEtFraction = nullptr;

  // PFMET variables
  MonitorElement *mPFphotonEtFraction = nullptr;
  MonitorElement *mPFphotonEt = nullptr;
  MonitorElement *mPFneutralHadronEtFraction = nullptr;
  MonitorElement *mPFneutralHadronEt = nullptr;
  MonitorElement *mPFelectronEtFraction = nullptr;
  MonitorElement *mPFelectronEt = nullptr;
  MonitorElement *mPFchargedHadronEtFraction = nullptr;
  MonitorElement *mPFchargedHadronEt = nullptr;
  MonitorElement *mPFmuonEtFraction = nullptr;
  MonitorElement *mPFmuonEt = nullptr;
  MonitorElement *mPFHFHadronEtFraction = nullptr;
  MonitorElement *mPFHFHadronEt = nullptr;
  MonitorElement *mPFHFEMEtFraction = nullptr;
  MonitorElement *mPFHFEMEt = nullptr;

  static constexpr double mEpsilon = 1e-6;

  // binnings: MET slices of the post-processor (multiples of 20 GeV) and phi slices
  // (integer values in [-2, 2]) must fall on bin edges
  static constexpr float phiEdge = 3.2;
  static constexpr float metEdge = 1100.;
  static constexpr float metDiffEdge = 600.;
  static constexpr float metRatioEdge = 5.;

  static constexpr int phiNbins = 32;        // 0.2 rad
  static constexpr int metNbins = 220;       // 5 GeV: post-processor MET slices (multiples of 20 GeV) fall on bin edges
  static constexpr int metCoarseNbins = 55;  // 20 GeV
  static constexpr int metDiffNbins = 300;   // 4 GeV
  static constexpr int metRatioNbins = 200;  // 0.025
};

#endif  // METTESTER_H
