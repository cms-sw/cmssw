#ifndef SimDataFormats_SimCluster_h
#define SimDataFormats_SimCluster_h

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include <vector>
#include <unordered_map>
#include <functional>
#include <ranges>

/** @brief Monte Carlo truth information used for calorimeter reco validation
 *
 * Object with copies to the original SimTrack, and eventual reference to GenParticle
 * Simulated calorimeter hits are saved as a list of pairs (DetId, fraction of reco hit energy contributed by the SimTrack)
 * ('absolute' hit energy is usually not saved)
 *
 * @author original author unknown, re-engineering and slimming by Subir Sarkar
 * (subir.sarkar@cern.ch), some tweaking and documentation by Mark Grimes
 * (mark.grimes@bristol.ac.uk).
 * @date original date unknown, re-engineering Jan-May 2013
 */
class SimCluster {
  friend std::ostream &operator<<(std::ostream &s, SimCluster const &tp);

public:
  typedef int Charge;                                       ///< electric charge type
  typedef math::XYZTLorentzVectorD LorentzVector;           ///< Lorentz vector
  typedef math::PtEtaPhiMLorentzVector PolarLorentzVector;  ///< Lorentz vector
  typedef math::XYZPointD Point;                            ///< point in the space
  typedef math::XYZVectorD Vector;                          ///< point in the space

  /// reference to reco::GenParticle
  typedef reco::GenParticleRefVector::iterator genp_iterator;
  typedef std::vector<SimTrack>::const_iterator g4t_iterator;

  SimCluster() = default;

  SimCluster(const SimTrack &simtrk);
  SimCluster(EncodedEventId eventID, uint32_t particleID);  // for PU
  /** Build a SimCluster from a collection of SimCluster. Hits&fractions are merged, SimTracks are all added in g4Tracks_ */
  template <typename R>
    requires std::ranges::input_range<R> && std::same_as<std::ranges::range_value_t<R>, SimCluster>
  static SimCluster mergeHitsFromCollection(R const &);

  /** @brief PDG ID.
   *
   * Returns the PDG ID of the first associated gen particle. If there are no
   * gen particles associated then it returns type() from the first SimTrack. */
  int pdgId() const {
    if (genParticles_.empty())
      return g4Tracks_[0].type();
    else
      return (*genParticles_.begin())->pdgId();
  }

  /** @brief Signal source, crossing number.
   *
   * Note this is taken from the first SimTrack only, but there shouldn't be any
   * SimTracks from different crossings in the SimCluster. */
  EncodedEventId eventId() const { return event_; }

  uint64_t particleId() const { return particleId_; }

  // Setters for G4 and reco::GenParticle
  void addGenParticle(const reco::GenParticleRef &ref) { genParticles_.push_back(ref); }
  void addG4Track(const SimTrack &t) { g4Tracks_.push_back(t); }
  /// iterators
  genp_iterator genParticle_begin() const { return genParticles_.begin(); }
  genp_iterator genParticle_end() const { return genParticles_.end(); }
  g4t_iterator g4Track_begin() const { return g4Tracks_.begin(); }
  g4t_iterator g4Track_end() const { return g4Tracks_.end(); }

  // Getters for Embd and Sim Tracks
  const reco::GenParticleRefVector &genParticles() const { return genParticles_; }
  // Only for clusters from the signal vertex
  const std::vector<SimTrack> &g4Tracks() const { return g4Tracks_; }

  /// @brief Electric charge. Note this is taken from the first SimTrack only.
  float charge() const { return g4Tracks_[0].charge(); }
  /// Gives charge in unit of quark charge (should be 3 times "charge()")
  int threeCharge() const { return lrintf(3.f * charge()); }

  /// @brief Four-momentum Lorentz vector. Note this is taken from the first
  /// SimTrack only.
  const math::XYZTLorentzVectorF &p4() const { return theMomentum_; }

  /// @brief spatial momentum vector
  math::XYZVectorF momentum() const { return p4().Vect(); }

  /// @brief Vector to boost to the particle centre of mass frame.
  math::XYZVectorF boostToCM() const { return p4().BoostToCM(); }

  /// @brief Magnitude of momentum vector. Note this is taken from the first
  /// SimTrack only.
  float p() const { return p4().P(); }

  /// @brief Energy. Note this is taken from the first SimTrack only.
  float energy() const { return p4().E(); }

  /// @brief Transverse energy. Note this is taken from the first SimTrack only.
  float et() const { return p4().Et(); }

  /// @brief Mass. Note this is taken from the first SimTrack only.
  float mass() const { return p4().M(); }

  /// @brief Mass squared. Note this is taken from the first SimTrack only.
  float massSqr() const { return pow(mass(), 2); }

  /// @brief Transverse mass. Note this is taken from the first SimTrack only.
  float mt() const { return p4().Mt(); }

  /// @brief Transverse mass squared. Note this is taken from the first SimTrack
  /// only.
  float mtSqr() const { return p4().Mt2(); }

  /// @brief x coordinate of momentum vector. Note this is taken from the first
  /// SimTrack only.
  float px() const { return p4().Px(); }

  /// @brief y coordinate of momentum vector. Note this is taken from the first
  /// SimTrack only.
  float py() const { return p4().Py(); }

  /// @brief z coordinate of momentum vector. Note this is taken from the first
  /// SimTrack only.
  float pz() const { return p4().Pz(); }

  /// @brief Transverse momentum. Note this is taken from the first SimTrack
  /// only.
  float pt() const { return p4().Pt(); }

  /// @brief Momentum azimuthal angle. Note this is taken from the first
  /// SimTrack only.
  float phi() const { return p4().Phi(); }

  /// @brief Momentum polar angle. Note this is taken from the first SimTrack
  /// only.
  float theta() const { return p4().Theta(); }

  /// @brief Momentum pseudorapidity. Note this is taken from the simtrack
  /// before the calorimeter
  float eta() const { return p4().Eta(); }

  /// @brief Rapidity. Note this is taken from the simtrack before the
  /// calorimeter
  float rapidity() const { return p4().Rapidity(); }

  /// @brief Same as rapidity().
  float y() const { return rapidity(); }

  /** @brief Status word.
   *
   * Returns status() from the first gen particle, or -99 if there are no gen
   * particles attached. */
  int status() const { return genParticles_.empty() ? -99 : (*genParticles_[0]).status(); }

  static const unsigned int longLivedTag;  ///< long lived flag

  /// is long lived?
  bool longLived() const { return status() & longLivedTag; }

  /** @brief Gives the total number of SimHits, in the cluster */
  int numberOfSimHits() const { return nsimhits_; }

  /** @brief Gives the total number of SimHits, in the cluster */
  int numberOfRecHits() const { return hits_.size(); }

  /** @brief add rechit with fraction */
  void addRecHitAndFraction(uint32_t hit, float fraction) {
    hits_.emplace_back(hit);
    fractions_.emplace_back(fraction);
  }

  /** @brief add rechit energy */
  void addHitEnergy(float energy) { energies_.emplace_back(energy); }

  /** @brief Returns list of rechit IDs and fractions for this SimCluster */
  std::vector<std::pair<uint32_t, float>> hits_and_fractions() const {
    std::vector<std::pair<uint32_t, float>> result;
    result.reserve(hits_.size());
    for (size_t i = 0; i < hits_.size(); ++i) {
      result.emplace_back(hits_[i], fractions_[i]);
    }
    return result;
  }

  /** @brief Returns filtered list of rechit IDs and fractions for this SimCluster based on a predicate */
  std::vector<std::pair<uint32_t, float>> filtered_hits_and_fractions(
      const std::function<bool(const DetId &)> &predicate) const {
    std::vector<std::pair<uint32_t, float>> result;
    for (size_t i = 0; i < hits_.size(); ++i) {
      DetId detid(hits_[i]);
      if (predicate(detid)) {
        result.emplace_back(hits_[i], fractions_[i]);
      }
    }
    return result;
  }

  /** @brief Returns list of rechit IDs and energies for this SimCluster */
  std::vector<std::pair<uint32_t, float>> hits_and_energies() const {
    assert(hits_.size() == energies_.size());
    std::vector<std::pair<uint32_t, float>> result;
    result.reserve(hits_.size());
    for (size_t i = 0; i < hits_.size(); ++i) {
      result.emplace_back(hits_[i], energies_[i]);
    }
    return result;
  }

  /** @brief clear the hits and fractions list */
  void clearHitsAndFractions() {
    std::vector<uint32_t>().swap(hits_);
    std::vector<float>().swap(fractions_);
  }

  /** @brief clear the energies list */
  void clearHitsEnergy() { std::vector<float>().swap(energies_); }

  /** @brief returns the accumulated sim energy in the cluster */
  float simEnergy() const { return simhit_energy_; }

  /** @brief add simhit's energy to cluster */
  void addSimHit(const PCaloHit &hit) {
    simhit_energy_ += hit.energy();
    ++nsimhits_;
  }

protected:
  uint64_t nsimhits_{0};
  EncodedEventId event_;

  uint32_t particleId_{0};
  float simhit_energy_{0.f};
  std::vector<uint32_t> hits_;
  std::vector<float> fractions_;
  std::vector<float> energies_;

  math::XYZTLorentzVectorF theMomentum_;

  std::vector<SimTrack> g4Tracks_;  ///< Copies of SimTrack used to build SimCluster (usually there is only one)
  /// Ref to GenParticle (in case the SimCluster is created from the entire GenParticle). Usually either empty or length 1
  reco::GenParticleRefVector genParticles_;
};

template <typename R>
  requires std::ranges::input_range<R> && std::same_as<std::ranges::range_value_t<R>, SimCluster>
SimCluster SimCluster::mergeHitsFromCollection(R const &inputs) {
  assert(!std::ranges::empty(inputs));
  SimCluster ret;
  ret.event_ = inputs[0].event_;
  ret.particleId_ = inputs[0].particleId_;

  ret.g4Tracks_.reserve(inputs.size());
  std::unordered_map<uint32_t, float> acc_fractions;  ///< Map DetId->(fraction)
  for (SimCluster const &other : inputs) {
    ret.simhit_energy_ += other.simhit_energy_;

    assert(other.hits_.size() == other.fractions_.size());
    for (std::size_t i = 0; i < other.hits_.size(); i++) {
      acc_fractions[other.hits_[i]] += other.fractions_[i];
    }

    ret.g4Tracks_.insert(ret.g4Tracks_.end(), other.g4Tracks_.begin(), other.g4Tracks_.end());
  }

  ret.hits_.reserve(acc_fractions.size());
  ret.fractions_.reserve(acc_fractions.size());
  for (const auto &hit_and_fraction : acc_fractions) {
    ret.hits_.push_back(hit_and_fraction.first);
    ret.fractions_.push_back(hit_and_fraction.second);
  }
  ret.nsimhits_ = ret.hits_.size();

  return ret;
}

#endif  // SimDataFormats_SimCluster_H
