#include <string>
#include <unordered_map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

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
  dqm::reco::MonitorElement* vect_sum_pt_;  // cumulative histogram
};

using Histograms_TICLPFValidation = std::unordered_map<int, Histogram_TICLPFValidation>;

class TICLPFValidation : public DQMGlobalEDAnalyzer<Histograms_TICLPFValidation> {
public:
  explicit TICLPFValidation(const edm::ParameterSet&);
  ~TICLPFValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      Histograms_TICLPFValidation&) const override;

  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms_TICLPFValidation const&) const override;

  // ----------member data ---------------------------
  std::string folder_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TICLPFValidation::TICLPFValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      pfCandidates_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("ticlPFCandidates"))) {
  //now do what ever initialization is needed
}

TICLPFValidation::~TICLPFValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------

void TICLPFValidation::dqmAnalyze(edm::Event const& iEvent,
                                  edm::EventSetup const& iSetup,
                                  Histograms_TICLPFValidation const& histos) const {
  using namespace edm;

  Handle<reco::PFCandidateCollection> pfCandidatesHandle;
  iEvent.getByToken(pfCandidates_, pfCandidatesHandle);
  reco::PFCandidateCollection const& pfCandidates = *pfCandidatesHandle;

  // pfCandidates
  double ptx_tot = 0.;
  double pty_tot = 0.;
  for (auto const pfc : pfCandidates) {
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
    ibook.setCurrentFolder(folder_ + "TICLPFCandidates/" + std::to_string(type));
    auto& histo = histos[type];
    histo.energy_ = ibook.book1D("Energy", "Energy", 250, 0., 250.);
    histo.pt_ = ibook.book1D("Pt", "Pt", 250, 0., 250.);
    histo.eta_ = ibook.book1D("Eta", "Eta", 100, -5., 5.);
    histo.phi_ = ibook.book1D("Phi", "Phi", 100, -4., 4.);
    histo.charge_ = ibook.book1D("Charge", "Charge", 3, -1.5, 1.5);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TICLPFValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "HGCAL/");  // Please keep the trailing '/'
  desc.add<edm::InputTag>("ticlPFCandidates", edm::InputTag("pfTICLProducer"));
  descriptions.add("ticlPFValidationDefault", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TICLPFValidation);
