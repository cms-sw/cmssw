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
  Decay chn1_;   //vector of allowed lepton decay channels for one W
  Decay chn2_;   //vector of allowed lepton decay channels for the other W
  Decay decay_;  //vector of allowed lepton decay channels for the other W
};
