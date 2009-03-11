#include "vector"
#include "string.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class TtDecayChannelSelector {
 public:
  enum Leaf
    {
      Elec=0,
      Muon=1,
      Tau =2
    };

  enum TauType
    {
      Leptonic=0,
      OneProng=1,
      ThreeProng=2
    };
    
  typedef std::vector<int> Decay;

  TtDecayChannelSelector(const edm::ParameterSet&);
  ~TtDecayChannelSelector();
  bool operator()(const reco::GenParticleCollection& parts, std::string inputType) const;

 private:

  bool isParticle(reco::GenParticleCollection::const_iterator& part, int pdgId, std::string& inputType) const;
  bool isParticle(reco::GenParticle::const_iterator& part, int pdgId, std::string& inputType) const;
  void parseDecayInput(Decay&, Decay&) const;
  void parseTauDecayInput(Decay&) const;
  unsigned int countChargedParticles(const reco::Candidate& part) const;
  bool checkTauDecay(const reco::Candidate&) const;

 private:
  std::string input_;
  bool  invert_;  //inversion flag
  int   channel_; //top decay channel
  int   summed_;
  Decay decay_;
  Decay chn1_;
  Decay chn2_;
  Decay tauDecay_;
};
