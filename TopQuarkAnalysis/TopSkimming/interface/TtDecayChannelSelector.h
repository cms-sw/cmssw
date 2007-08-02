#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

class TtDecayChannelSelector {
 public:
  enum Leaf
    {
      Elec=0,
      Muon=1,
      Tau =2
    };
  typedef std::vector<int> Decay;

  TtDecayChannelSelector(const edm::ParameterSet&);
  ~TtDecayChannelSelector();
  bool operator()(const reco::CandidateCollection&) const;

 private:

  void parseDecayInput(Decay&, Decay&) const;

 private:
  bool  invert_;  //inversion flag
  int   channel_; //top decay channel
  int   summed_;
  Decay decay_;
};
