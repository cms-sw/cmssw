#include <memory>
#include <string>
#include <vector>
#include <map>

#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"

class StDecaySubset : public TopDecaySubset {
 public:

  explicit StDecaySubset(const edm::ParameterSet&);
  ~StDecaySubset();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  /// fill output vector with full decay chain (for single top like generator listing)
  void fillSingleTopOutput(const reco::GenParticleCollection&, reco::GenParticleCollection&);
};
