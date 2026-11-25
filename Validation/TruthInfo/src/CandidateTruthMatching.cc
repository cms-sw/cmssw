#include "Validation/TruthInfo/interface/CandidateTruthMatching.h"

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <unordered_set>

namespace truth::candidate {

  ParticleClass truthClass(int32_t pdgId, double charge) {
    const int32_t a = std::abs(pdgId);
    if (a == 22 || a == 111)
      return ParticleClass::Photon;
    if (a == 11)
      return ParticleClass::Electron;
    if (a == 13)
      return ParticleClass::Muon;
    if (a >= 100)
      return charge != 0. ? ParticleClass::ChargedHadron : ParticleClass::NeutralHadron;
    return ParticleClass::Other;
  }

  ParticleClass candidateClass(int pdgId) {
    const int a = std::abs(pdgId);
    if (a == 22 || a == 111)
      return ParticleClass::Photon;
    if (a == 11)
      return ParticleClass::Electron;
    if (a == 13)
      return ParticleClass::Muon;
    if (a == 211 || a == 321 || a == 2212)
      return ParticleClass::ChargedHadron;
    if (a >= 100)
      return ParticleClass::NeutralHadron;
    return ParticleClass::Other;
  }

  LevelProjection::LevelProjection(Graph const& graph, std::span<const unsigned int> members)
      : memberSlot_(graph.nParticles(), kNoMember), above_(graph.nParticles(), 0) {
    const uint32_t nParticles = graph.nParticles();
    const uint32_t nVertices = graph.nVertices();
    std::vector<uint32_t> stack;
    stack.reserve(members.size());
    for (uint32_t slot = 0; slot < members.size(); ++slot) {
      const uint32_t m = members[slot];
      if (m < nParticles && memberSlot_[m] == kNoMember) {
        memberSlot_[m] = slot;
        stack.push_back(m);
      }
    }
    // Down: a descendant takes the slot of the first member that reaches it, so the walk visits a particle once.
    for (std::size_t head = 0; head < stack.size(); ++head) {
      const uint32_t p = stack[head];
      for (const uint32_t v : graph.decayVertices(p)) {
        if (v >= nVertices)
          continue;
        for (const uint32_t child : graph.outgoingParticles(v)) {
          if (child < nParticles && memberSlot_[child] == kNoMember) {
            memberSlot_[child] = memberSlot_[p];
            stack.push_back(child);
          }
        }
      }
    }
    // Up: every strict ancestor of a member.
    stack.clear();
    for (const unsigned int m : members) {
      if (m < nParticles)
        stack.push_back(m);
    }
    while (!stack.empty()) {
      const uint32_t p = stack.back();
      stack.pop_back();
      for (const uint32_t v : graph.productionVertices(p)) {
        if (v >= nVertices)
          continue;
        for (const uint32_t parent : graph.incomingParticles(v)) {
          if (parent < nParticles && parent != p && above_[parent] == 0) {
            above_[parent] = 1;
            stack.push_back(parent);
          }
        }
      }
    }
  }

  bool hasAncestorIn(Graph const& graph, uint32_t particle, std::span<const uint32_t> sortedIds) {
    if (sortedIds.empty())
      return false;
    const uint32_t nParticles = graph.nParticles();
    const uint32_t nVertices = graph.nVertices();
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> stack{particle};
    while (!stack.empty()) {
      const uint32_t p = stack.back();
      stack.pop_back();
      if (p >= nParticles)
        continue;
      for (const uint32_t v : graph.productionVertices(p)) {
        if (v >= nVertices)
          continue;
        for (const uint32_t parent : graph.incomingParticles(v)) {
          if (parent == p || !seen.insert(parent).second)
            continue;
          if (std::binary_search(sortedIds.begin(), sortedIds.end(), parent))
            return true;
          stack.push_back(parent);
        }
      }
    }
    return false;
  }

  MemberOutcome evaluateMember(ParticleClass memberClass,
                               double trueEnergy,
                               std::span<const MemberCandidate> candidates,
                               LadderConfig const& config) {
    MemberOutcome out;
    if (candidates.empty())
      return out;
    out.reconstructed = true;
    out.multiple = candidates.size() > 1;
    // For a charged member a candidate with a track of the member first; then the largest shared energy, then the
    // first candidate.
    const bool charged = isCharged(memberClass);
    std::size_t rep = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      out.energySum += candidates[i].energy;
      auto const& c = candidates[i];
      auto const& best = candidates[rep];
      if (charged && c.trackOnMember != best.trackOnMember) {
        if (c.trackOnMember)
          rep = i;
      } else if (c.sharedEnergyFraction > best.sharedEnergyFraction) {
        rep = i;
      }
    }
    out.representative = static_cast<int>(rep);
    auto const& r = candidates[rep];
    out.trackCorrect = charged ? (r.hasTrack && r.trackOnMember) : !r.hasTrack;
    out.pidCorrect = out.trackCorrect && r.candidateClass == memberClass;
    const double response = trueEnergy > 0. ? r.energy / trueEnergy : -1.;
    out.energyCorrect = out.pidCorrect && response >= config.energyResponseMin && response <= config.energyResponseMax;
    return out;
  }

  double leadingShare(LevelProjection const& dominance,
                      std::span<const RowMatch> referenceRow,
                      bool hasTrackster,
                      uint32_t trackParticle) {
    if (referenceRow.empty())
      return !hasTrackster && dominance.memberSlot(trackParticle) != LevelProjection::kNoMember ? 1. : -1.;
    // A branch and its descendants add up as one contributor, the member they project on.
    std::vector<std::pair<uint32_t, double>> contributions;
    contributions.reserve(referenceRow.size());
    double total = 0.;
    for (auto const& match : referenceRow) {
      const uint32_t member = dominance.memberSlot(match.particle);
      if (member == LevelProjection::kNoMember)
        continue;
      total += match.sharedEnergy;
      auto it = std::find_if(
          contributions.begin(), contributions.end(), [member](auto const& p) { return p.first == member; });
      if (it == contributions.end())
        contributions.emplace_back(member, match.sharedEnergy);
      else
        it->second += match.sharedEnergy;
    }
    if (total <= 0.)
      return -1.;
    double best = 0.;
    for (auto const& contribution : contributions)
      best = std::max(best, contribution.second);
    return best / total;
  }

  CandidateOutcome classifyCandidate(Graph const& graph,
                                     LevelProjection const& dominance,
                                     DominanceConfig const& config,
                                     CandidateInput const& candidate) {
    const uint32_t nParticles = graph.nParticles();
    CandidateOutcome out;
    out.calo = candidate.caloIdentity < nParticles;
    const uint32_t trackParticle = candidate.trackParticle < nParticles ? candidate.trackParticle : kNoParticle;
    if (out.calo)
      out.identity = candidate.caloIdentity;
    else if (!candidate.hasTrackster)
      out.identity = trackParticle;
    out.associated = out.identity != kNoParticle;
    const uint32_t idMember = out.associated ? dominance.memberSlot(out.identity) : LevelProjection::kNoMember;
    out.levelAssoc = idMember != LevelProjection::kNoMember;
    out.merged = out.associated && !out.levelAssoc && dominance.isAbove(out.identity);
    out.pileup = out.associated && graph.particles()[out.identity].isFromPileup();
    out.hasLevelCandidate = candidate.leadingShare >= 0.;
    out.dominated = out.associated && (!out.hasLevelCandidate || candidate.leadingShare >= config.minLeadingTruthShare);
    out.purity = out.calo ? 1. - candidate.caloScore : (out.associated ? candidate.trackPurity : 0.);
    out.trackCalo = trackParticle != kNoParticle && out.calo;
    // The track and the calorimeter part are one object: the same particle, the same dominance-level member, or a
    // track inside the merged object the calorimeter part matched.
    out.trackConsistent =
        out.trackCalo &&
        (trackParticle == out.identity || (out.levelAssoc && dominance.memberSlot(trackParticle) == idMember) ||
         (out.merged && hasAncestorIn(graph, trackParticle, std::span<const uint32_t>(&out.identity, 1))));
    const bool memberKnown = out.levelAssoc && idMember < config.memberClass.size();
    out.pidCorrect = memberKnown && config.memberClass[idMember] == candidate.candidateClass;
    if (memberKnown)
      out.identityBin = static_cast<std::size_t>(config.memberClass[idMember]);
    else if (out.merged)
      out.identityBin = kIdentityMerged;
    else if (out.pileup)
      out.identityBin = kIdentityPileup;
    else if (out.associated)
      out.identityBin = kIdentityOutsideLevel;
    return out;
  }

}  // namespace truth::candidate
