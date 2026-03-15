#ifndef SimDataFormats_SimCluster_h
#define SimDataFormats_SimCluster_h

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <functional>
#include <iosfwd>
#include <limits>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

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

  // Zero-copy hit+fraction view (iterates as pairs)
  // Not used directly; use the named derived structs below.
  struct HitsAndValuesViewBase {
    std::span<const uint32_t> hits;
    std::span<const float> values;

    struct iterator {
      using iterator_category = std::random_access_iterator_tag;
      using value_type = std::pair<uint32_t, float>;
      using difference_type = std::ptrdiff_t;
      using reference = value_type;  // returned by value
      using pointer = void;

      const uint32_t *h = nullptr;
      const float *v = nullptr;

      reference operator*() const { return {*h, *v}; }

      iterator &operator++() {
        ++h;
        ++v;
        return *this;
      }
      iterator operator++(int) {
        auto tmp = *this;
        ++(*this);
        return tmp;
      }
      iterator &operator--() {
        --h;
        --v;
        return *this;
      }
      iterator operator--(int) {
        auto tmp = *this;
        --(*this);
        return tmp;
      }

      iterator &operator+=(difference_type n) {
        h += n;
        v += n;
        return *this;
      }
      iterator &operator-=(difference_type n) { return (*this) += (-n); }

      friend iterator operator+(iterator it, difference_type n) { return it += n; }
      friend iterator operator-(iterator it, difference_type n) { return it -= n; }
      friend difference_type operator-(iterator a, iterator b) { return a.h - b.h; }

      friend bool operator==(iterator a, iterator b) { return a.h == b.h; }
      friend bool operator!=(iterator a, iterator b) { return !(a == b); }
      friend bool operator<(iterator a, iterator b) { return a.h < b.h; }
    };

    iterator begin() const { return iterator{hits.data(), values.data()}; }
    iterator end() const {
      return iterator{hits.data() + static_cast<std::ptrdiff_t>(hits.size()),
                      values.data() + static_cast<std::ptrdiff_t>(values.size())};
    }

    iterator::value_type operator[](size_t n) const {
      assert(n < values.size());
      return {hits[n], values[n]};
    }

    size_t size() const { return hits.size(); }
    bool empty() const { return hits.empty(); }
  };

  struct HitsAndFractionsView : HitsAndValuesViewBase {
    std::span<const float> &fractions = values;
  };
  struct HitsAndEnergiesView : HitsAndValuesViewBase {
    std::span<const float> &energies = values;
  };

  SimCluster() = default;

  SimCluster(const SimTrack &simtrk) {
    g4Tracks_.push_back(simtrk);
    theMomentum_.SetPxPyPzE(
        simtrk.momentum().px(), simtrk.momentum().py(), simtrk.momentum().pz(), simtrk.momentum().E());
    event_ = simtrk.eventId();
    particleId_ = simtrk.trackId();
  }

  SimCluster(EncodedEventId eventID, uint32_t particleID) : event_(eventID), particleId_(particleID) {}

  ~SimCluster() = default;

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
    hitsFinalized_ = false;
  }

  /** @brief add rechit energy */
  void addHitEnergy(float energy) {
    energies_.emplace_back(energy);
    hitsFinalized_ = false;
  }

  /** @brief Returns list of rechit IDs and fractions for this SimCluster (copying legacy API) */
  std::vector<std::pair<uint32_t, float>> hits_and_fractions() const {
    // legacy returns a copy; now deterministic because finalizeHits() sorted it
    std::vector<std::pair<uint32_t, float>> result;
    result.reserve(hits_.size());
	for (size_t i = 0; i < hits_.size(); ++i) {
      result.emplace_back(hits_[i], fractions_[i]);
    }
    return result;
  }

  /** @brief Returns filtered list of rechit IDs and fractions for this SimCluster based on a predicate (copying) */
  std::vector<std::pair<uint32_t, float>> filtered_hits_and_fractions(
      const std::function<bool(const DetId &)> &predicate) const {
    std::vector<std::pair<uint32_t, float>> result;
    for (size_t i = 0; i < hits_.size(); ++i) {
      DetId detid(hits_[i]);
      if (predicate(detid))
        result.emplace_back(hits_[i], fractions_[i]);
    }
    return result;
  }

  /** @brief Returns list of rechit IDs and energies for this SimCluster (copying legacy API) */
  std::vector<std::pair<uint32_t, float>> hits_and_energies() const {
    assert(hits_.size() == energies_.size());
    std::vector<std::pair<uint32_t, float>> result;
    result.reserve(hits_.size());
    for (size_t i = 0; i < hits_.size(); ++i)
      result.emplace_back(hits_[i], energies_[i]);
    return result;
  }

  void clearHitsAndFractions() {
    std::vector<uint32_t>().swap(hits_);
    std::vector<float>().swap(fractions_);
    hitsFinalized_ = false;
  }

  void clearHitsEnergy() {
    std::vector<float>().swap(energies_);
    hitsFinalized_ = false;
  }

  /** @brief returns the accumulated sim energy in the cluster */
  float simEnergy() const { return simhit_energy_; }

  /** @brief add simhit's energy to cluster */
  void addSimHit(const PCaloHit &hit) {
    simhit_energy_ += hit.energy();
    ++nsimhits_;
  }

  // --------------------------------------------------------------------------
  // New: producer-side "finalization" (to be called before putting in the event)
  // --------------------------------------------------------------------------
  void finalizeHits() {
    // Keep your original implicit invariant:
    assert(hits_.size() == fractions_.size() && !hits_.empty());
    // Energies are optional but if present must align.
    if (!energies_.empty())
      assert(energies_.size() == hits_.size());

    // Already finalized? keep it cheap and idempotent.
    if (hitsFinalized_)
      return;

    // Sort by (det, subdet, rawid)
    std::vector<size_t> order(hits_.size());
    std::iota(order.begin(), order.end(), 0);

    auto key = [&](size_t i) {
      DetId id(hits_[i]);
      return std::tuple<int, int, uint32_t>(static_cast<int>(id.det()), id.subdetId(), hits_[i]);
    };

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return key(a) < key(b); });

    applyPermutation_(hits_, order);
    applyPermutation_(fractions_, order);
    if (!energies_.empty())
      applyPermutation_(energies_, order);

    buildDetRanges_();

    hitsFinalized_ = true;
  }

  // --------------------------------------------------------------------------
  // cost-free views (requires a call to finalizeHits())
  // --------------------------------------------------------------------------
  HitsAndFractionsView hits_and_fractions_view() const {
    assertFinalized_();
    return HitsAndFractionsView{{hits_, fractions_}};
  }

  HitsAndFractionsView hits_and_fractions_view(DetId::Detector det) const {
    assertFinalized_();
    auto [b, e] = detRanges_[detIndex_(det)];
    return HitsAndFractionsView{{{hits_.data() + b, e - b}, {fractions_.data() + b, e - b}}};
  }

  HitsAndFractionsView hits_and_fractions_view(DetId::Detector det, int subdetId) const {
    assertFinalized_();
    auto [bb, ee] = subdetRange_(det, subdetId);
    return HitsAndFractionsView{{{hits_.data() + bb, ee - bb}, {fractions_.data() + bb, ee - bb}}};
  }

  HitsAndFractionsView hits_and_fractions_view(DetId::Detector detIdMin, DetId::Detector detIdMax) const {
    assertFinalized_();
    auto [begin, end] = detMinMaxRange_(detIdMin, detIdMax);
    return HitsAndFractionsView{{{hits_.data() + begin, end - begin}, {fractions_.data() + begin, end - begin}}};
  }

  HitsAndEnergiesView hits_and_energies_view() const {
    assertFinalized_();
    assertEnergies_();
    return HitsAndEnergiesView{{hits_, energies_}};
  }

  HitsAndEnergiesView hits_and_energies_view(DetId::Detector det) const {
    assertFinalized_();
    assertEnergies_();
    auto [b, e] = detRanges_[detIndex_(det)];
    return HitsAndEnergiesView{{{hits_.data() + b, e - b}, {energies_.data() + b, e - b}}};
  }

  HitsAndEnergiesView hits_and_energies_view(DetId::Detector det, int subdetId) const {
    assertFinalized_();
    assertEnergies_();
    auto [bb, ee] = subdetRange_(det, subdetId);
    return HitsAndEnergiesView{{{hits_.data() + bb, ee - bb}, {energies_.data() + bb, ee - bb}}};
  }

  HitsAndEnergiesView hits_and_energies_view(DetId::Detector detIdMin, DetId::Detector detIdMax) const {
    assertFinalized_();
    assertEnergies_();
    auto [begin, end] = detMinMaxRange_(detIdMin, detIdMax);
    return HitsAndEnergiesView{{{hits_.data() + begin, end - begin}, {energies_.data() + begin, end - begin}}};
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

  /// references to G4 and reco::GenParticle tracks
  std::vector<SimTrack> g4Tracks_;
  reco::GenParticleRefVector genParticles_;

private:
  static constexpr size_t kMaxDetectors_ = 32;  // Probably 16 could be enough

  bool hitsFinalized_{false};
  std::array<std::pair<size_t, size_t>, kMaxDetectors_> detRanges_{};  // [begin,end) per detector

  static size_t detIndex_(DetId::Detector det) {
    const auto idx = static_cast<size_t>(det);
    assert(idx < kMaxDetectors_);
    return idx;
  }

  void assertFinalized_() const {
    assert(hitsFinalized_ && "SimCluster: hits not finalized. Call finalizeHits() in the producer before persisting.");
  }

  void assertEnergies_() const {
    assert(!energies_.empty() &&
           "SimCluster: energies is empty; populate it with addHitEnergy() before calling this view.");
  }

  // Returns the [begin, end) index range within hits_ for a det+subdet pair.
  std::pair<size_t, size_t> subdetRange_(DetId::Detector det, int subdetId) const {
    auto [b, e] = detRanges_[detIndex_(det)];
    if (b == e)
      return {b, e};

    auto beginIt = hits_.begin() + static_cast<std::ptrdiff_t>(b);
    auto endIt = hits_.begin() + static_cast<std::ptrdiff_t>(e);

    auto keyOf = [](uint32_t rawid) {
      DetId id(rawid);
      return std::pair<int, uint32_t>(id.subdetId(), rawid);
    };

    const auto lowKey = std::pair<int, uint32_t>(subdetId, 0u);
    const auto highKey = std::pair<int, uint32_t>(subdetId, std::numeric_limits<uint32_t>::max());

    auto lo = std::lower_bound(beginIt, endIt, lowKey, [&](uint32_t rawid, const auto &k) { return keyOf(rawid) < k; });
    auto hi =
        std::upper_bound(beginIt, endIt, highKey, [&](const auto &k, uint32_t rawid) { return k < keyOf(rawid); });

    return {static_cast<size_t>(std::distance(hits_.begin(), lo)),
            static_cast<size_t>(std::distance(hits_.begin(), hi))};
  }

  // Returns the [begin, end) index range spanning all detectors in [detIdMin, detIdMax].
  std::pair<size_t, size_t> detMinMaxRange_(DetId::Detector detIdMin, DetId::Detector detIdMax) const {
    const auto detIndexMin = detIndex_(detIdMin);
    const auto detIndexMax = detIndex_(detIdMax);
    assert(detIndexMin <= detIndexMax);
    const auto [bMin, eMin] = detRanges_[detIndexMin];
    const auto [bMax, eMax] = detRanges_[detIndexMax];
    return {bMin, eMax};
  }

  void buildDetRanges_() {
    detRanges_.fill({0u, 0u});
    size_t i = 0;
    while (i < hits_.size()) {
      DetId id(hits_[i]);
      const auto idx = detIndex_(static_cast<DetId::Detector>(id.det()));
      const size_t begin = i;
      do {
        ++i;
      } while (i < hits_.size() && DetId(hits_[i]).det() == id.det());
      detRanges_[idx] = {begin, i};
    }
  }

  template <typename T>
  static void applyPermutation_(std::vector<T> &v, const std::vector<size_t> &order) {
    std::vector<T> tmp;
    tmp.reserve(v.size());
    for (size_t idx : order)
      tmp.push_back(v[idx]);
    v.swap(tmp);
  }
};

#endif  // SimDataFormats_SimCluster_H
