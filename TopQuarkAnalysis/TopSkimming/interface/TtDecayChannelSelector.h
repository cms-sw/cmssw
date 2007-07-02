#include <vector>

//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

class TtDecayChannelSelector {
 public:
  TtDecayChannelSelector(const edm::ParameterSet&);
  ~TtDecayChannelSelector();
  bool operator()(const reco::CandidateCollection&) const;
 private:
  int  topChn_;
  bool invChn_;
  bool subChn_;
  bool invSub_;
  std::vector<int> lepVec_;
};
