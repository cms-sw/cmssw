#include <cmath>
#include <string>
#include <unordered_map>

// user include files
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

struct Histogram_TICLPFValidation {
  dqm::reco::MonitorElement* type_;
  dqm::reco::MonitorElement* energy_;
  dqm::reco::MonitorElement* pt_;
  dqm::reco::MonitorElement* eta_;
  dqm::reco::MonitorElement* phi_;
  dqm::reco::MonitorElement* charge_;
  dqm::reco::MonitorElement* logPtVsEta_;
  dqm::reco::MonitorElement* vect_sum_pt_;  // cumulative histogram
};

using Histograms_TICLPFValidation = std::unordered_map<int, Histogram_TICLPFValidation>;

class TICLPFValidation : public DQMGlobalEDAnalyzer<Histograms_TICLPFValidation> {
public:
  explicit TICLPFValidation(const edm::ParameterSet&);
  ~TICLPFValidation() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      Histograms_TICLPFValidation&) const override;

  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms_TICLPFValidation const&) const override;

  // ----------member data ---------------------------
  const std::string folder_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
  static constexpr std::array<std::string, reco::PFCandidate::egamma_HF + 1> kPFCandidateTypeNames_ = {
      "X", "h", "e", "mu", "gamma", "h0", "h_HF", "egamma_HF"};

  static constexpr std::array<std::string, reco::PFCandidate::egamma_HF + 1> kPFCandidateDisplayNames = {
      "Undefined", "ChHadron", "Electron", "Muon", "Photon", "NHadron", "HFHadron", "HFEGamma"};
};

TICLPFValidation::TICLPFValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      pfCandidates_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("ticlPFCandidates"))) {}

//
// member functions
//

// ------------ method called for each event  ------------

void TICLPFValidation::dqmAnalyze(edm::Event const& iEvent,
                                  edm::EventSetup const& iSetup,
                                  Histograms_TICLPFValidation const& histos) const {
  using namespace edm;

  const auto& pfCandidatesHandle = iEvent.getHandle(pfCandidates_);
  if (!pfCandidatesHandle.isValid()) {
    edm::LogWarning("TICLPFValidation") << "Invalid PFCandidateCollection handle, skipping event.";
    return;
  }
  reco::PFCandidateCollection const& pfCandidates = *pfCandidatesHandle;

  // pfCandidates
  double ptx_tot = 0.;
  double pty_tot = 0.;
  for (auto const& pfc : pfCandidates) {
    size_t type = pfc.particleId();
    ptx_tot += pfc.px();
    pty_tot += pfc.py();
    histos.at(0).type_->Fill(type);
    auto& histo = histos.at(type);
    histo.energy_->Fill(pfc.energy());
    histo.pt_->Fill(pfc.pt());
    histo.eta_->Fill(pfc.eta());
    histo.phi_->Fill(pfc.phi());
    histo.charge_->Fill(pfc.charge());
    if (pfc.pt() > 0.)
      histo.logPtVsEta_->Fill(pfc.eta(), std::log10(pfc.pt()));
  }
  auto& histo = histos.at(0);
  histo.vect_sum_pt_->Fill(std::sqrt(ptx_tot * ptx_tot + pty_tot * pty_tot));
}

void TICLPFValidation::bookHistograms(DQMStore::IBooker& ibook,
                                      edm::Run const& run,
                                      edm::EventSetup const& iSetup,
                                      Histograms_TICLPFValidation& histos) const {
  ibook.setCurrentFolder(folder_ + "TICLPFCandidates/");
  histos[0].type_ = ibook.book1D("Type", "Type", 10, -0.5, 9.5);
  histos[0].vect_sum_pt_ = ibook.book1D("PtVectSum", "PtVectSum", 200, 0., 200.);
  for (size_t type = reco::PFCandidate::X; type <= reco::PFCandidate::egamma_HF; type++) {
    ibook.setCurrentFolder(folder_ + "TICLPFCandidates/" + kPFCandidateTypeNames_[type]);
    auto& histo = histos[type];

    const auto& particleType = kPFCandidateDisplayNames[type];

    histo.energy_ = ibook.book1D("Energy", particleType + " Energy", 250, 0., 250.);
    histo.energy_->setAxisTitle("Energy [GeV]", 1);

    histo.pt_ = ibook.book1D("Pt", particleType + " p_{T}", 250, 0., 250.);
    histo.pt_->setAxisTitle("p_{T} [GeV]", 1);

    histo.eta_ = ibook.book1D("Eta", particleType + " #eta", 100, -5., 5.);
    histo.eta_->setAxisTitle("#eta", 1);

    histo.phi_ = ibook.book1D("Phi", particleType + " #phi", 100, -M_PI, M_PI);
    histo.phi_->setAxisTitle("#phi", 1);

    histo.charge_ = ibook.book1D("Charge", particleType + " Charge", 3, -1.5, 1.5);
    histo.charge_->setAxisTitle("Charge", 1);

    histo.logPtVsEta_ = ibook.book2D("LogPtVsEta", particleType + " log(p_{T}) vs #eta", 100, -5., 5., 80, -1., 3.);
    histo.logPtVsEta_->setAxisTitle("#eta", 1);
    histo.logPtVsEta_->setAxisTitle("log(p_{T} / GeV)", 2);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TICLPFValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "HGCAL/");  // Please keep the trailing '/'
  desc.add<edm::InputTag>("ticlPFCandidates", edm::InputTag("pfTICL"));
  descriptions.add("ticlPFValidationDefault", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TICLPFValidation);
