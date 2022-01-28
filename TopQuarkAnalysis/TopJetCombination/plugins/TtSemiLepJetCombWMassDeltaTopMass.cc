#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "TopQuarkAnalysis/TopTools/interface/MEzCalculator.h"

class TtSemiLepJetCombWMassDeltaTopMass : public edm::global::EDProducer<> {
public:
  explicit TtSemiLepJetCombWMassDeltaTopMass(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& setup) const override;

  bool isValid(const int& idx, const std::vector<pat::Jet>& jets) const {
    return (0 <= idx && idx < (int)jets.size());
  };

  edm::EDGetTokenT<std::vector<pat::Jet>> jetsToken_;
  edm::EDGetTokenT<edm::View<reco::RecoCandidate>> lepsToken_;
  edm::EDGetTokenT<std::vector<pat::MET>> metsToken_;
  int maxNJets_;
  double wMass_;
  bool useBTagging_;
  std::string bTagAlgorithm_;
  double minBDiscBJets_;
  double maxBDiscLightJets_;
  int neutrinoSolutionType_;
};

TtSemiLepJetCombWMassDeltaTopMass::TtSemiLepJetCombWMassDeltaTopMass(const edm::ParameterSet& cfg)
    : jetsToken_(consumes<std::vector<pat::Jet>>(cfg.getParameter<edm::InputTag>("jets"))),
      lepsToken_(consumes<edm::View<reco::RecoCandidate>>(cfg.getParameter<edm::InputTag>("leps"))),
      metsToken_(consumes<std::vector<pat::MET>>(cfg.getParameter<edm::InputTag>("mets"))),
      maxNJets_(cfg.getParameter<int>("maxNJets")),
      wMass_(cfg.getParameter<double>("wMass")),
      useBTagging_(cfg.getParameter<bool>("useBTagging")),
      bTagAlgorithm_(cfg.getParameter<std::string>("bTagAlgorithm")),
      minBDiscBJets_(cfg.getParameter<double>("minBDiscBJets")),
      maxBDiscLightJets_(cfg.getParameter<double>("maxBDiscLightJets")),
      neutrinoSolutionType_(cfg.getParameter<int>("neutrinoSolutionType")) {
  if (maxNJets_ < 4 && maxNJets_ != -1)
    throw cms::Exception("WrongConfig") << "Parameter maxNJets can not be set to " << maxNJets_ << ". \n"
                                        << "It has to be larger than 4 or can be set to -1 to take all jets.";

  produces<std::vector<std::vector<int>>>();
  produces<int>("NumberOfConsideredJets");
}

void TtSemiLepJetCombWMassDeltaTopMass::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& setup) const {
  auto pOut = std::make_unique<std::vector<std::vector<int>>>();
  auto pJetsConsidered = std::make_unique<int>(0);

  std::vector<int> match;
  for (unsigned int i = 0; i < 4; ++i)
    match.push_back(-1);

  // get jets
  const auto& jets = evt.get(jetsToken_);

  // get leptons
  const auto& leps = evt.get(lepsToken_);

  // get MET
  const auto& mets = evt.get(metsToken_);

  // skip events without lepton candidate or less than 4 jets or no MET
  if (leps.empty() || jets.size() < 4 || mets.empty()) {
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
  // associate those jets that get closest to the W mass
  // with their invariant mass to the hadronic W boson
  // -----------------------------------------------------
  double wDist = -1.;
  std::vector<int> closestToWMassIndices;
  closestToWMassIndices.push_back(-1);
  closestToWMassIndices.push_back(-1);
  for (unsigned idx = 0; idx < maxNJets; ++idx) {
    if (useBTagging_ && (!isLJet[idx] || (cntBJets <= 2 && isBJet[idx])))
      continue;
    for (unsigned jdx = (idx + 1); jdx < maxNJets; ++jdx) {
      if (useBTagging_ &&
          (!isLJet[jdx] || (cntBJets <= 2 && isBJet[jdx]) || (cntBJets == 3 && isBJet[idx] && isBJet[jdx])))
        continue;
      reco::Particle::LorentzVector sum = jets[idx].p4() + jets[jdx].p4();
      if (wDist < 0. || wDist > fabs(sum.mass() - wMass_)) {
        wDist = fabs(sum.mass() - wMass_);
        closestToWMassIndices.clear();
        closestToWMassIndices.push_back(idx);
        closestToWMassIndices.push_back(jdx);
      }
    }
  }

  // -----------------------------------------------------
  // build a leptonic W boson from the lepton and the MET
  // -----------------------------------------------------
  double neutrino_pz = 0.;
  if (neutrinoSolutionType_ != -1) {
    MEzCalculator mez;
    mez.SetMET(*mets.begin());
    if (dynamic_cast<const reco::Muon*>(&(leps.front())))
      mez.SetLepton(leps[0], true);
    else if (dynamic_cast<const reco::GsfElectron*>(&(leps.front())))
      mez.SetLepton(leps[0], false);
    else
      throw cms::Exception("UnimplementedFeature")
          << "Type of lepton given together with MET for solving neutrino kinematics is neither muon nor electron.\n";
    neutrino_pz = mez.Calculate(neutrinoSolutionType_);
  }
  const math::XYZTLorentzVector neutrino(mets.begin()->px(),
                                         mets.begin()->py(),
                                         neutrino_pz,
                                         sqrt(mets.begin()->px() * mets.begin()->px() +
                                              mets.begin()->py() * mets.begin()->py() + neutrino_pz * neutrino_pz));
  const reco::Particle::LorentzVector lepW = neutrino + leps.front().p4();

  // -----------------------------------------------------
  // associate those jets to the hadronic and the leptonic
  // b quark that minimize the difference between
  // hadronic and leptonic top mass
  // -----------------------------------------------------
  double deltaTop = -1.;
  int hadB = -1;
  int lepB = -1;
  if (isValid(closestToWMassIndices[0], jets) && isValid(closestToWMassIndices[1], jets)) {
    const reco::Particle::LorentzVector hadW =
        jets[closestToWMassIndices[0]].p4() + jets[closestToWMassIndices[1]].p4();
    // find hadronic b candidate
    for (unsigned idx = 0; idx < maxNJets; ++idx) {
      if (useBTagging_ && !isBJet[idx])
        continue;
      // make sure it's not used up already from the hadronic W
      if ((int)idx != closestToWMassIndices[0] && (int)idx != closestToWMassIndices[1]) {
        reco::Particle::LorentzVector hadTop = hadW + jets[idx].p4();
        // find leptonic b candidate
        for (unsigned jdx = 0; jdx < maxNJets; ++jdx) {
          if (useBTagging_ && !isBJet[jdx])
            continue;
          // make sure it's not used up already from the hadronic branch
          if ((int)jdx != closestToWMassIndices[0] && (int)jdx != closestToWMassIndices[1] && jdx != idx) {
            reco::Particle::LorentzVector lepTop = lepW + jets[jdx].p4();
            if (deltaTop < 0. || deltaTop > fabs(hadTop.mass() - lepTop.mass())) {
              deltaTop = fabs(hadTop.mass() - lepTop.mass());
              hadB = idx;
              lepB = jdx;
            }
          }
        }
      }
    }
  }

  match[TtSemiLepEvtPartons::LightQ] = closestToWMassIndices[0];
  match[TtSemiLepEvtPartons::LightQBar] = closestToWMassIndices[1];
  match[TtSemiLepEvtPartons::HadB] = hadB;
  match[TtSemiLepEvtPartons::LepB] = lepB;

  pOut->push_back(match);
  evt.put(std::move(pOut));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtSemiLepJetCombWMassDeltaTopMass);
