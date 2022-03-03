#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "Math/VectorUtil.h"

class TtSemiLepJetCombGeom : public edm::global::EDProducer<> {
public:
  explicit TtSemiLepJetCombGeom(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& setup) const override;

  bool isValid(const int idx, const std::vector<pat::Jet>& jets) const { return (0 <= idx && idx < (int)jets.size()); };
  double distance(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&) const;

  edm::EDGetTokenT<std::vector<pat::Jet>> jetsToken_;
  edm::EDGetTokenT<edm::View<reco::RecoCandidate>> lepsToken_;
  int maxNJets_;
  bool useDeltaR_;
  bool useBTagging_;
  std::string bTagAlgorithm_;
  double minBDiscBJets_;
  double maxBDiscLightJets_;
};

TtSemiLepJetCombGeom::TtSemiLepJetCombGeom(const edm::ParameterSet& cfg)
    : jetsToken_(consumes<std::vector<pat::Jet>>(cfg.getParameter<edm::InputTag>("jets"))),
      lepsToken_(consumes<edm::View<reco::RecoCandidate>>(cfg.getParameter<edm::InputTag>("leps"))),
      maxNJets_(cfg.getParameter<int>("maxNJets")),
      useDeltaR_(cfg.getParameter<bool>("useDeltaR")),
      useBTagging_(cfg.getParameter<bool>("useBTagging")),
      bTagAlgorithm_(cfg.getParameter<std::string>("bTagAlgorithm")),
      minBDiscBJets_(cfg.getParameter<double>("minBDiscBJets")),
      maxBDiscLightJets_(cfg.getParameter<double>("maxBDiscLightJets")) {
  if (maxNJets_ < 4 && maxNJets_ != -1)
    throw cms::Exception("WrongConfig") << "Parameter maxNJets can not be set to " << maxNJets_ << ". \n"
                                        << "It has to be larger than 4 or can be set to -1 to take all jets.";

  produces<std::vector<std::vector<int>>>();
  produces<int>("NumberOfConsideredJets");
}

void TtSemiLepJetCombGeom::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& setup) const {
  auto pOut = std::make_unique<std::vector<std::vector<int>>>();
  auto pJetsConsidered = std::make_unique<int>(0);

  std::vector<int> match;
  for (unsigned int i = 0; i < 4; ++i)
    match.push_back(-1);

  // get jets
  const auto& jets = evt.get(jetsToken_);

  // get leptons
  const auto& leps = evt.get(lepsToken_);

  // skip events without lepton candidate or less than 4 jets
  if (leps.empty() || jets.size() < 4) {
    pOut->push_back(match);
    evt.put(std::move(pOut));
    *pJetsConsidered = jets.size();
    evt.put(std::move(pJetsConsidered), "NumberOfConsideredJets");
    return;
  }

  unsigned maxNJets = maxNJets_;
  if (maxNJets_ == -1 || (int)jets.size() < maxNJets_)
    maxNJets = jets.size();
  *pJetsConsidered = maxNJets;
  evt.put(std::move(pJetsConsidered), "NumberOfConsideredJets");

  std::vector<bool> isBJet;
  std::vector<bool> isLJet;
  int cntBJets = 0;
  if (useBTagging_) {
    for (unsigned int idx = 0; idx < maxNJets; ++idx) {
      isBJet.push_back((jets[idx].bDiscriminator(bTagAlgorithm_) > minBDiscBJets_));
      isLJet.push_back((jets[idx].bDiscriminator(bTagAlgorithm_) < maxBDiscLightJets_));
      if (jets[idx].bDiscriminator(bTagAlgorithm_) > minBDiscBJets_)
        cntBJets++;
    }
  }

  // -----------------------------------------------------
  // associate those two jets to the hadronic W boson that
  // have the smallest distance to each other
  // -----------------------------------------------------
  double minDist = -1.;
  int lightQ = -1;
  int lightQBar = -1;
  for (unsigned int idx = 0; idx < maxNJets; ++idx) {
    if (useBTagging_ && (!isLJet[idx] || (cntBJets <= 2 && isBJet[idx])))
      continue;
    for (unsigned int jdx = (idx + 1); jdx < maxNJets; ++jdx) {
      if (useBTagging_ &&
          (!isLJet[jdx] || (cntBJets <= 2 && isBJet[jdx]) || (cntBJets == 3 && isBJet[idx] && isBJet[jdx])))
        continue;
      double dist = distance(jets[idx].p4(), jets[jdx].p4());
      if (minDist < 0. || dist < minDist) {
        minDist = dist;
        lightQ = idx;
        lightQBar = jdx;
      }
    }
  }

  reco::Particle::LorentzVector wHad;
  if (isValid(lightQ, jets) && isValid(lightQBar, jets))
    wHad = jets[lightQ].p4() + jets[lightQBar].p4();

  // -----------------------------------------------------
  // associate to the hadronic b quark the remaining jet
  // that has the smallest distance to the hadronic W
  // -----------------------------------------------------
  minDist = -1.;
  int hadB = -1;
  if (isValid(lightQ, jets) && isValid(lightQBar, jets)) {
    for (unsigned int idx = 0; idx < maxNJets; ++idx) {
      if (useBTagging_ && !isBJet[idx])
        continue;
      // make sure it's not used up already from the hadronic W
      if ((int)idx != lightQ && (int)idx != lightQBar) {
        double dist = distance(jets[idx].p4(), wHad);
        if (minDist < 0. || dist < minDist) {
          minDist = dist;
          hadB = idx;
        }
      }
    }
  }

  // -----------------------------------------------------
  // associate to the leptonic b quark the remaining jet
  // that has the smallest distance to the leading lepton
  // -----------------------------------------------------
  minDist = -1.;
  int lepB = -1;
  for (unsigned int idx = 0; idx < maxNJets; ++idx) {
    if (useBTagging_ && !isBJet[idx])
      continue;
    // make sure it's not used up already from the hadronic decay chain
    if ((int)idx != lightQ && (int)idx != lightQBar && (int)idx != hadB) {
      double dist = distance(jets[idx].p4(), leps[0].p4());
      if (minDist < 0. || dist < minDist) {
        minDist = dist;
        lepB = idx;
      }
    }
  }

  match[TtSemiLepEvtPartons::LightQ] = lightQ;
  match[TtSemiLepEvtPartons::LightQBar] = lightQBar;
  match[TtSemiLepEvtPartons::HadB] = hadB;
  match[TtSemiLepEvtPartons::LepB] = lepB;

  pOut->push_back(match);
  evt.put(std::move(pOut));
}

double TtSemiLepJetCombGeom::distance(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2) const {
  // calculate the distance between two lorentz vectors
  // using DeltaR or DeltaTheta
  if (useDeltaR_)
    return ROOT::Math::VectorUtil::DeltaR(v1, v2);
  return fabs(v1.theta() - v2.theta());
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtSemiLepJetCombGeom);
