
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"
using namespace edm;

void GenHIEvent::setGenParticles(const reco::GenParticleCollection* input) {
  subevents_.reserve(nhard_);
  for (int i = 0; i < nhard_; ++i) {
    std::vector<reco::GenParticleRef> refs;
    subevents_.push_back(refs);
  }

  for (unsigned int i = 0; i < input->size(); ++i) {
    reco::GenParticleRef ref(input, i);
    subevents_[ref->collisionId()].push_back(ref);
  }
}

const std::vector<reco::GenParticleRef> GenHIEvent::getSubEvent(unsigned int sub_id) const {
  if (sub_id > subevents_.size()) {  // sub_id >= 0, since sub_id is unsigned
  }

  return subevents_[sub_id];
}
