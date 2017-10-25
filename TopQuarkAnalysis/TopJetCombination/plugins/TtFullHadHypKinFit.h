#ifndef TtFullHadHypKinFit_h
#define TtFullHadHypKinFit_h

#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullHadHypothesis.h"

#include "DataFormats/PatCandidates/interface/Particle.h"

class TtFullHadHypKinFit : public TtFullHadHypothesis  {

 public:

  explicit TtFullHadHypKinFit(const edm::ParameterSet&);
  ~TtFullHadHypKinFit() override;

 private:

  /// build the event hypothesis key
  void buildKey() override { key_= TtFullHadronicEvent::kKinFit; };
  /// build event hypothesis from the reco objects of a full-hadronic event
  void buildHypo(edm::Event&,
			 const edm::Handle<std::vector<pat::Jet> >&,
			 std::vector<int>&, const unsigned int iComb) override;

  edm::EDGetTokenT<std::vector<int> > statusToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightQToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightQBarToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > bToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > bBarToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightPToken_;
  edm::EDGetTokenT<std::vector<pat::Particle> > lightPBarToken_;

};

#endif
