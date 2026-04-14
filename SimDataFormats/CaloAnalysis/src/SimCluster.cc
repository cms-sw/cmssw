#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include <numeric>
#include <ostream>

const unsigned int SimCluster::longLivedTag = 65536;

SimCluster::SimCluster(const SimTrack &simtrk) {
  addG4Track(simtrk);
  event_ = simtrk.eventId();
  particleId_ = simtrk.trackId();

  theMomentum_.SetPxPyPzE(
      simtrk.momentum().px(), simtrk.momentum().py(), simtrk.momentum().pz(), simtrk.momentum().E());
}

SimCluster::SimCluster(EncodedEventId eventID, uint32_t particleID) {
  event_ = eventID;
  particleId_ = particleID;
}

SimCluster SimCluster::mergeHitsFromCollection(const std::vector<SimCluster> &inputs) {
  assert(!inputs.empty());
  SimCluster ret;
  ret.event_ = inputs[0].event_;
  ret.particleId_ = inputs[0].particleId_;

  ret.g4Tracks_.reserve(inputs.size());
  std::unordered_map<uint32_t, float> acc_energies, acc_fractions;
  for (SimCluster const &other : inputs) {
    ret.simhit_energy_ += other.simhit_energy_;

    assert(other.hits_.size() == other.energies_.size());
    for (std::size_t i = 0; i < other.hits_.size(); i++) {
      acc_energies[other.hits_[i]] += other.energies_[i];
    }

    if (!other.fractions_.empty()) {
      assert(other.hits_.size() == other.fractions_.size());
      for (std::size_t i = 0; i < other.hits_.size(); i++) {
        acc_fractions[other.hits_[i]] += other.fractions_[i];
      }
    }

    ret.g4Tracks_.insert(ret.g4Tracks_.end(), other.g4Tracks_.begin(), other.g4Tracks_.end());
  }

  ret.hits_.reserve(acc_energies.size());
  ret.energies_.reserve(acc_energies.size());
  ret.fractions_.reserve(acc_fractions.size());
  for (auto const &[detId, energy] : acc_energies) {
    ret.hits_.push_back(detId);
    ret.energies_.push_back(energy);
  }
  for (auto const &[_, fraction] : acc_fractions) {
    ret.fractions_.push_back(fraction);
  }
  ret.nsimhits_ = ret.hits_.size();

  return ret;
}

std::ostream &operator<<(std::ostream &s, SimCluster const &tp) {
  s << "CP momentum, q, ID, & Event #: " << tp.p4() << " " << tp.charge() << " " << tp.pdgId() << " "
    << tp.eventId().bunchCrossing() << "." << tp.eventId().event() << std::endl;

  for (SimCluster::genp_iterator hepT = tp.genParticle_begin(); hepT != tp.genParticle_end(); ++hepT) {
    s << " HepMC Track Momentum " << (*hepT)->momentum().rho() << std::endl;
  }

  for (SimCluster::g4t_iterator g4T = tp.g4Track_begin(); g4T != tp.g4Track_end(); ++g4T) {
    s << " Geant Track Momentum  " << g4T->momentum() << std::endl;
    s << " Geant Track ID & type " << g4T->trackId() << " " << g4T->type() << std::endl;
    if (g4T->type() != tp.pdgId()) {
      s << " Mismatch b/t SimCluster and Geant types" << std::endl;
    }
  }
  s << " # of cells = " << tp.hits_.size()
    << ", effective cells = " << std::accumulate(tp.fractions_.begin(), tp.fractions_.end(), 0.f) << std::endl;
  return s;
}
