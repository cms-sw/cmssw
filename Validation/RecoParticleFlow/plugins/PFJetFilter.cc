#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"
#include <atomic>

class PFJetFilter : public edm::global::EDFilter<> {
public:
  explicit PFJetFilter(const edm::ParameterSet &);
  ~PFJetFilter() override;

private:
  void beginJob() override;
  bool filter(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  void endJob() override;

  double resolution(double, bool) const;
  double response(double, bool) const;

  double recPt_cut;
  double genPt_cut;
  double deltaEt_min;
  double deltaEt_max;
  double deltaR_min;
  double deltaR_max;
  double eta_min;
  double eta_max;
  edm::EDGetTokenT<edm::View<reco::Candidate>> inputTruthLabel_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> inputRecoLabel_;

  mutable std::atomic<unsigned int> entry;
  bool verbose;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetFilter);

using namespace reco;
using namespace edm;
using namespace std;

PFJetFilter::PFJetFilter(const edm::ParameterSet &iConfig) {
  inputTruthLabel_ = consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("InputTruthLabel"));
  inputRecoLabel_ = consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("InputRecoLabel"));

  recPt_cut = iConfig.getParameter<double>("recPt");
  genPt_cut = iConfig.getParameter<double>("genPt");

  eta_min = iConfig.getParameter<double>("minEta");
  eta_max = iConfig.getParameter<double>("maxEta");

  deltaR_min = iConfig.getParameter<double>("deltaRMin");
  deltaR_max = iConfig.getParameter<double>("deltaRMax");

  deltaEt_min = iConfig.getParameter<double>("minDeltaEt");
  deltaEt_max = iConfig.getParameter<double>("maxDeltaEt");

  verbose = iConfig.getParameter<bool>("verbose");

  entry = 0;
}

PFJetFilter::~PFJetFilter() {}

void PFJetFilter::beginJob() {}

void PFJetFilter::endJob() {}

bool PFJetFilter::filter(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iESetup) const {
  // Typedefs to use views
  typedef edm::View<reco::Candidate> candidateCollection;
  typedef edm::View<reco::Candidate> candidateCollection;

  const candidateCollection *truth_candidates;
  const candidateCollection *reco_candidates;

  // ==========================================================
  // Retrieve!
  // ==========================================================

  // Get Truth Candidates (GenCandidates, GenJets, etc.)
  Handle<candidateCollection> truth_hnd;
  bool isGen = iEvent.getByToken(inputTruthLabel_, truth_hnd);
  if (!isGen) {
    std::cout << "Warning : no Gen jets in input !" << std::endl;
    return false;
  }

  truth_candidates = truth_hnd.product();

  // Get Reco Candidates (PFlow, CaloJet, etc.)
  Handle<candidateCollection> reco_hnd;
  bool isReco = iEvent.getByToken(inputRecoLabel_, reco_hnd);
  if (!isReco) {
    std::cout << "Warning : no Reco jets in input !" << std::endl;
    return false;
  }
  reco_candidates = reco_hnd.product();
  if (!truth_candidates || !reco_candidates)
    return false;

  bool pass = false;

  for (unsigned int i = 0; i < reco_candidates->size(); i++) {
    const reco::Candidate *particle = &(*reco_candidates)[i];

    // This protection is meant at not being used !
    assert(particle != nullptr);

    double rec_pt = particle->pt();
    double rec_eta = particle->eta();
    double rec_phi = particle->phi();

    // skip PFjets with pt < recPt_cut GeV
    if (rec_pt < recPt_cut)
      continue;

    // skip PFjets with eta > maxEta_cut or eta < minEta_cut
    if (fabs(rec_eta) > eta_max)
      continue;
    if (fabs(rec_eta) < eta_min)
      continue;

    bool Barrel = false;
    bool Endcap = false;
    if (std::abs(rec_eta) < 1.4)
      Barrel = true;
    if (std::abs(rec_eta) > 1.4 && std::abs(rec_eta) < 2.6)
      Endcap = true;

    // Keep only jets in the barrel or the endcaps, within the tracker
    // acceptance
    if (!Barrel && !Endcap)
      continue;

    // Find the closets recJet
    double deltaRmin = 999.;
    double ptmin = 0.;
    for (unsigned int j = 0; j < reco_candidates->size(); j++) {
      if (i == j)
        continue;
      const reco::Candidate *other = &(*reco_candidates)[j];
      double deltaR = PFBenchmarkAlgo::deltaR(particle, other);
      if (deltaR < deltaRmin && other->pt() > 0.25 * particle->pt() && other->pt() > recPt_cut) {
        deltaRmin = deltaR;
        ptmin = other->pt();
      }
      if (deltaRmin < deltaR_min)
        break;
    }
    if (deltaRmin < deltaR_min)
      continue;

    // Find the closest genJet.
    const reco::Candidate *gen_particle = PFBenchmarkAlgo::matchByDeltaR(particle, truth_candidates);

    // Check there is a genJet associated to the recoJet
    if (gen_particle == nullptr)
      continue;

    // check deltaR is small enough
    double deltaR = PFBenchmarkAlgo::deltaR(particle, gen_particle);
    if (deltaR > deltaR_max)
      continue;

    // double true_E = gen_particle->p();
    double true_pt = gen_particle->pt();
    double true_eta = gen_particle->eta();
    double true_phi = gen_particle->phi();

    // skip PFjets with pt < genPt_cut GeV
    if (true_pt < genPt_cut)
      continue;

    // Find the pT residual
    double resPt = (rec_pt - true_pt) / true_pt;
    double sigma = resolution(true_pt, Barrel);
    double avera = response(true_pt, Barrel);
    double nSig = (resPt - avera) / sigma;

    if (nSig > deltaEt_max || nSig < deltaEt_min) {
      /* */
      if (verbose)
        std::cout << "Entry " << entry++ << " resPt = " << resPt << " sigma/avera/nSig = " << sigma << "/" << avera
                  << "/" << nSig << " pT (T/R) " << true_pt << "/" << rec_pt << " Eta (T/R) " << true_eta << "/"
                  << rec_eta << " Phi (T/R) " << true_phi << "/" << rec_phi << " deltaRMin/ptmin " << deltaRmin << "/"
                  << ptmin << std::endl;
      /* */
      pass = true;
    }

    if (pass)
      break;
  }

  return pass;
}

double PFJetFilter::resolution(double pt, bool barrel) const {
  double p0 = barrel ? 1.19200e-02 : 8.45341e-03;
  double p1 = barrel ? 1.06138e+00 : 7.96855e-01;
  double p2 = barrel ? -2.05929e+00 : -3.12076e-01;

  double resp = p0 + p1 / sqrt(pt) + p2 / pt;
  return resp;
}

double PFJetFilter::response(double pt, bool barrel) const {
  double p0 = barrel ? 1.09906E-1 : 6.91939E+1;
  double p1 = barrel ? -1.61443E-1 : -6.92733E+1;
  double p2 = barrel ? 3.45489E+3 : 1.58207E+6;

  double resp = p0 + p1 * std::exp(-pt / p2);
  return resp;
}
