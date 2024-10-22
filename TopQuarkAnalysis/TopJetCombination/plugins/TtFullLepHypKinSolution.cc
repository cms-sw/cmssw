#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullLepHypothesis.h"
#include "DataFormats/PatCandidates/interface/Particle.h"

class TtFullLepHypKinSolution : public TtFullLepHypothesis {
public:
  explicit TtFullLepHypKinSolution(const edm::ParameterSet&);

private:
  /// build the event hypothesis key
  void buildKey() override { key_ = TtEvent::kKinSolution; };
  /// build event hypothesis from the reco objects of a full-leptonic event
  void buildHypo(edm::Event& evt,
                 const edm::Handle<std::vector<pat::Electron> >& elecs,
                 const edm::Handle<std::vector<pat::Muon> >& mus,
                 const edm::Handle<std::vector<pat::Jet> >& jets,
                 const edm::Handle<std::vector<pat::MET> >& mets,
                 std::vector<int>& match,
                 const unsigned int iComb) override;

  //   edm::EDGetTokenT<std::vector<std::vector<int> > > particleIdcsToken_;
  edm::EDGetTokenT<std::vector<reco::LeafCandidate> > nusToken_;
  edm::EDGetTokenT<std::vector<reco::LeafCandidate> > nuBarsToken_;
  edm::EDGetTokenT<std::vector<double> > solWeightToken_;
};

TtFullLepHypKinSolution::TtFullLepHypKinSolution(const edm::ParameterSet& cfg)
    : TtFullLepHypothesis(cfg),
      nusToken_(consumes<std::vector<reco::LeafCandidate> >(cfg.getParameter<edm::InputTag>("Neutrinos"))),
      nuBarsToken_(consumes<std::vector<reco::LeafCandidate> >(cfg.getParameter<edm::InputTag>("NeutrinoBars"))),
      solWeightToken_(consumes<std::vector<double> >(cfg.getParameter<edm::InputTag>("solutionWeight"))) {}

void TtFullLepHypKinSolution::buildHypo(edm::Event& evt,
                                        const edm::Handle<std::vector<pat::Electron> >& elecs,
                                        const edm::Handle<std::vector<pat::Muon> >& mus,
                                        const edm::Handle<std::vector<pat::Jet> >& jets,
                                        const edm::Handle<std::vector<pat::MET> >& mets,
                                        std::vector<int>& match,
                                        const unsigned int iComb) {
  edm::Handle<std::vector<double> > solWeight;
  //   edm::Handle<std::vector<std::vector<int> > >   idcsVec;
  edm::Handle<std::vector<reco::LeafCandidate> > nus;
  edm::Handle<std::vector<reco::LeafCandidate> > nuBars;

  evt.getByToken(solWeightToken_, solWeight);
  //   evt.getByToken(particleIdcsToken_, idcsVec  );
  evt.getByToken(nusToken_, nus);
  evt.getByToken(nuBarsToken_, nuBars);

  if ((*solWeight)[iComb] < 0) {
    // create empty hypothesis if no solution exists
    return;
  }

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if (!jets->empty()) {
    b_ = makeCandidate(jets, match[0], jetCorrectionLevel_);
    bBar_ = makeCandidate(jets, match[1], jetCorrectionLevel_);
  }
  // -----------------------------------------------------
  // add leptons
  // -----------------------------------------------------
  if (!elecs->empty() && match[2] >= 0)
    leptonBar_ = makeCandidate(elecs, match[2]);

  if (!elecs->empty() && match[3] >= 0)
    lepton_ = makeCandidate(elecs, match[3]);

  if (!mus->empty() && match[4] >= 0 && match[2] < 0)
    leptonBar_ = makeCandidate(mus, match[4]);

  // this 'else' happens if you have a wrong charge electron-muon-
  // solution so the indices are (b-idx, bbar-idx, 0, -1, 0, -1)
  // so the mu^+ is stored as l^-
  else if (!mus->empty() && match[4] >= 0)
    lepton_ = makeCandidate(mus, match[4]);

  if (!mus->empty() && match[5] >= 0 && match[3] < 0)
    lepton_ = makeCandidate(mus, match[5]);

  // this 'else' happens if you have a wrong charge electron-muon-
  // solution so the indices are (b-idx, bbar-idx, -1, 0, -1, 0)
  // so the mu^- is stored as l^+
  else if (!mus->empty() && match[5] >= 0)
    leptonBar_ = makeCandidate(mus, match[5]);

  // -----------------------------------------------------
  // add neutrinos
  // -----------------------------------------------------
  if (!nus->empty())
    neutrino_ = makeCandidate(nus, iComb);

  if (!nuBars->empty())
    neutrinoBar_ = makeCandidate(nuBars, iComb);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtFullLepHypKinSolution);
