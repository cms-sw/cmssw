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
  int channel_;  //top decay channel
  bool invert_;  //inversion flag
  Decay oneLep_; //vector of considered decay channels with one lepton
  Decay twoLep_; //vector of considered decay channels with two leptons
};
