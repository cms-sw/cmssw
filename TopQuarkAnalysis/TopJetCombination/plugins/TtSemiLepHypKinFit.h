#ifndef TtSemiLepHypKinFit_h
#define TtSemiLepHypKinFit_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"

#include "DataFormats/PatCandidates/interface/Particle.h"

class TtSemiLepHypKinFit : public TtSemiLepHypothesis  {

 public:

  explicit TtSemiLepHypKinFit(const edm::ParameterSet&);
  ~TtSemiLepHypKinFit() override;

 private:

  /// build the event hypothesis key
  void buildKey() override { key_= TtSemiLeptonicEvent::kKinFit; };
  /// build event hypothesis from the reco objects of a semi-leptonic event
  void buildHypo(edm::Event&,
			 const edm::Handle<edm::View<reco::RecoCandidate> >&,
			 const edm::Handle<std::vector<pat::MET> >&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb) override;

  edm::EDGetTokenT<std::vector<int> > statusToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsHadPToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsHadQToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsHadBToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > partonsLepBToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > leptonsToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > neutrinosToken_;

};

#endif
