#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <catch2/catch_all.hpp>

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "Validation/TruthInfo/interface/CandidateTruthMatching.h"

using namespace truth::candidate;

namespace {
  void csr(uint32_t n,
           std::vector<std::pair<uint32_t, uint32_t>> pairs,
           std::vector<uint32_t>& offsets,
           std::vector<uint32_t>& flat) {
    std::sort(pairs.begin(), pairs.end());
    offsets.assign(n + 1, 0);
    for (auto const& pr : pairs)
      ++offsets[pr.first + 1];
    for (uint32_t i = 1; i <= n; ++i)
      offsets[i] += offsets[i - 1];
    flat.assign(pairs.size(), 0);
    auto cursor = offsets;
    for (auto const& pr : pairs)
      flat[cursor[pr.first]++] = pr.second;
  }

  // p0 tau decays at v0 to p1 pi+, p2 pi0 and p3 nu; p2 decays at v1 to the photons p4 and p5; p1 interacts at v2
  // and makes p6; p7 is an unrelated pion.
  constexpr uint32_t kTau = 0, kPion = 1, kPi0 = 2, kNu = 3, kPhoton1 = 4, kPhoton2 = 5, kSecondary = 6, kOther = 7;

  truth::Graph tauGraph() {
    truth::Graph g;
    g.particles().resize(8);
    g.vertices().resize(3);
    const std::vector<std::pair<uint32_t, uint32_t>> decays{{kTau, 0}, {kPi0, 1}, {kPion, 2}};
    const std::vector<std::pair<uint32_t, uint32_t>> productions{
        {kPion, 0}, {kPi0, 0}, {kNu, 0}, {kPhoton1, 1}, {kPhoton2, 1}, {kSecondary, 2}};
    std::vector<std::pair<uint32_t, uint32_t>> incoming, outgoing;
    for (auto const& [p, v] : decays)
      incoming.emplace_back(v, p);
    for (auto const& [p, v] : productions)
      outgoing.emplace_back(v, p);
    csr(8, decays, g.particleToDecayVertexOffsets(), g.particleToDecayVertices());
    csr(8, productions, g.particleToProductionVertexOffsets(), g.particleToProductionVertices());
    csr(3, outgoing, g.vertexToOutgoingParticleOffsets(), g.vertexToOutgoingParticles());
    csr(3, incoming, g.vertexToIncomingParticleOffsets(), g.vertexToIncomingParticles());
    return g;
  }
}  // namespace

TEST_CASE("A particle projects on the member it descends from", "[CandidateTruthMatching]") {
  const auto g = tauGraph();
  REQUIRE(g.isConsistent());

  SECTION("the pi0 is a member") {
    const std::vector<unsigned int> members{kPion, kPi0};
    const LevelProjection projection(g, members);
    CHECK(projection.memberSlot(kPion) == 0);
    CHECK(projection.memberSlot(kSecondary) == 0);
    CHECK(projection.memberSlot(kPi0) == 1);
    CHECK(projection.memberSlot(kPhoton2) == 1);
    CHECK(projection.memberSlot(kTau) == LevelProjection::kNoMember);
    CHECK(projection.isAbove(kTau));
    CHECK_FALSE(projection.isAbove(kPi0));
    CHECK(projection.memberSlot(kOther) == LevelProjection::kNoMember);
    CHECK_FALSE(projection.isAbove(kOther));
  }

  SECTION("the photons are the members") {
    const std::vector<unsigned int> members{kPion, kPhoton1, kPhoton2};
    const LevelProjection projection(g, members);
    CHECK(projection.memberSlot(kPhoton2) == 2);
    CHECK(projection.memberSlot(kPi0) == LevelProjection::kNoMember);
    CHECK(projection.isAbove(kPi0));
    CHECK(projection.isAbove(kTau));
    CHECK_FALSE(projection.isAbove(kNu));
  }
}

TEST_CASE("An ancestor is found among sorted ids", "[CandidateTruthMatching]") {
  const auto g = tauGraph();
  const std::vector<uint32_t> pi0{kPi0}, pion{kPion}, tauAndOther{kTau, kOther};
  CHECK(hasAncestorIn(g, kPhoton1, pi0));
  CHECK_FALSE(hasAncestorIn(g, kPhoton1, pion));
  CHECK(hasAncestorIn(g, kPhoton1, tauAndOther));
  CHECK_FALSE(hasAncestorIn(g, kPi0, pi0));
  CHECK_FALSE(hasAncestorIn(g, kOther, tauAndOther));
}

TEST_CASE("Truth and candidate classes", "[CandidateTruthMatching]") {
  CHECK(truthClass(111, 0.) == ParticleClass::Photon);
  CHECK(truthClass(22, 0.) == ParticleClass::Photon);
  CHECK(truthClass(-11, 1.) == ParticleClass::Electron);
  CHECK(truthClass(13, -1.) == ParticleClass::Muon);
  CHECK(truthClass(-211, -1.) == ParticleClass::ChargedHadron);
  CHECK(truthClass(130, 0.) == ParticleClass::NeutralHadron);
  CHECK(truthClass(2112, 0.) == ParticleClass::NeutralHadron);
  CHECK(truthClass(15, -1.) == ParticleClass::Other);
  CHECK(candidateClass(-211) == ParticleClass::ChargedHadron);
  CHECK(candidateClass(130) == ParticleClass::NeutralHadron);
  CHECK(candidateClass(22) == ParticleClass::Photon);
  CHECK(candidateClass(11) == ParticleClass::Electron);
  CHECK(candidateClass(-13) == ParticleClass::Muon);
  CHECK(candidateClass(0) == ParticleClass::Other);
}

TEST_CASE("The ladder is read on the representative candidate", "[CandidateTruthMatching]") {
  const LadderConfig ladder{0.7, 1.3};

  SECTION("no candidate") {
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., {}, ladder);
    CHECK_FALSE(out.reconstructed);
    CHECK(out.representative == -1);
  }

  SECTION("a charged hadron with its own track, class and energy") {
    const std::vector<MemberCandidate> cands{{0.8f, ParticleClass::ChargedHadron, true, true, 9.}};
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., cands, ladder);
    CHECK(out.reconstructed);
    CHECK_FALSE(out.multiple);
    CHECK(out.trackCorrect);
    CHECK(out.pidCorrect);
    CHECK(out.energyCorrect);
  }

  SECTION("a charged hadron with the track of another particle") {
    const std::vector<MemberCandidate> cands{{0.8f, ParticleClass::ChargedHadron, true, false, 9.}};
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., cands, ladder);
    CHECK(out.reconstructed);
    CHECK_FALSE(out.trackCorrect);
    CHECK_FALSE(out.pidCorrect);
  }

  SECTION("a photon with a track") {
    const std::vector<MemberCandidate> cands{{0.9f, ParticleClass::Electron, true, false, 10.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK_FALSE(out.trackCorrect);
  }

  SECTION("a photon with the class of a neutral hadron") {
    const std::vector<MemberCandidate> cands{{0.9f, ParticleClass::NeutralHadron, false, false, 10.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK(out.trackCorrect);
    CHECK_FALSE(out.pidCorrect);
  }

  SECTION("a member with two candidates reads the candidate that shares most of it") {
    const std::vector<MemberCandidate> cands{{0.3f, ParticleClass::Photon, false, false, 3.},
                                             {0.6f, ParticleClass::Photon, false, false, 6.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK(out.multiple);
    CHECK(out.representative == 1);
    CHECK(out.pidCorrect);
    CHECK_FALSE(out.energyCorrect);
    CHECK(out.energySum == 9.);
  }

  SECTION("for a charged member the candidate with its track is the representative") {
    // A track-only charged candidate and the neutral candidate that took the trackster.
    const std::vector<MemberCandidate> cands{{0.9f, ParticleClass::NeutralHadron, false, false, 4.},
                                             {0.f, ParticleClass::ChargedHadron, true, true, 10.}};
    const auto out = evaluateMember(ParticleClass::ChargedHadron, 10., cands, ladder);
    CHECK(out.representative == 1);
    CHECK(out.multiple);
    CHECK(out.energyCorrect);
  }

  SECTION("for a neutral member the largest share is the representative, whatever the tracks") {
    // A photon that converted: an electron candidate with a leg track and a photon candidate.
    const std::vector<MemberCandidate> cands{{0.3f, ParticleClass::Electron, true, true, 3.},
                                             {0.6f, ParticleClass::Photon, false, false, 7.}};
    const auto out = evaluateMember(ParticleClass::Photon, 10., cands, ladder);
    CHECK(out.representative == 1);
    CHECK(out.pidCorrect);
    CHECK(out.energyCorrect);
  }
}

TEST_CASE("The leading share is read on the reference row", "[CandidateTruthMatching]") {
  const auto g = tauGraph();
  const std::vector<unsigned int> members{kPion, kPhoton1, kPhoton2};
  const LevelProjection dominance(g, members);

  SECTION("a branch and its descendants add up as one member") {
    const std::vector<RowMatch> row{{kPhoton1, 3.f}, {kSecondary, 1.f}, {kPion, 1.f}, {kTau, 5.f}};
    CHECK(leadingShare(dominance, row, true, kNoParticle) == Catch::Approx(0.6));
  }

  SECTION("an empty reference row") {
    const std::vector<RowMatch> empty;
    CHECK(leadingShare(dominance, empty, true, kPion) == -1.);
    CHECK(leadingShare(dominance, empty, false, kOther) == -1.);
    CHECK(leadingShare(dominance, empty, false, kNoParticle) == -1.);
    CHECK(leadingShare(dominance, empty, false, kPion) == 1.);
  }

  SECTION("a row with no member of the level") {
    const std::vector<RowMatch> row{{kTau, 5.f}, {kOther, 1.f}};
    CHECK(leadingShare(dominance, row, true, kPion) == -1.);
  }
}

TEST_CASE("A candidate is classified against its identity", "[CandidateTruthMatching]") {
  auto g = tauGraph();
  // kOther is from a pileup interaction.
  g.particles()[kOther].eventId = EncodedEventId(0, 1).rawId();
  const std::vector<unsigned int> members{kPion, kPhoton1, kPhoton2};
  const LevelProjection dominance(g, members);
  const std::vector<ParticleClass> memberClass{
      ParticleClass::ChargedHadron, ParticleClass::Photon, ParticleClass::Photon};
  const DominanceConfig config{memberClass, 0.5};

  SECTION("a track-only candidate takes the particle of its track") {
    const CandidateInput in{.candidateClass = ParticleClass::ChargedHadron,
                            .hasTrackster = false,
                            .hasTrack = true,
                            .trackParticle = kPion,
                            .trackPurity = 0.9f,
                            .leadingShare = 1.};
    const auto out = classifyCandidate(g, dominance, config, in);
    CHECK(out.identity == kPion);
    CHECK(out.associated);
    CHECK_FALSE(out.calo);
    CHECK(out.levelAssoc);
    CHECK(out.dominated);
    CHECK(out.purity == Catch::Approx(0.9));
    CHECK_FALSE(out.trackCalo);
    CHECK(out.pidCorrect);
    CHECK(out.identityBin == static_cast<std::size_t>(ParticleClass::ChargedHadron));
  }

  SECTION("a candidate with a trackster and no calorimeter match is not associated through its track") {
    const CandidateInput in{.candidateClass = ParticleClass::ChargedHadron,
                            .hasTrackster = true,
                            .hasTrack = true,
                            .trackParticle = kPion,
                            .trackPurity = 0.9f,
                            .leadingShare = -1.};
    const auto out = classifyCandidate(g, dominance, config, in);
    CHECK_FALSE(out.associated);
    CHECK_FALSE(out.dominated);
    CHECK_FALSE(out.hasLevelCandidate);
    CHECK(out.purity == 0.);
    CHECK(out.identityBin == kIdentityNone);
  }

  SECTION("a merged calorimeter identity with the track inside it") {
    const CandidateInput in{.candidateClass = ParticleClass::Electron,
                            .hasTrackster = true,
                            .caloIdentity = kPi0,
                            .caloScore = 0.2f,
                            .hasTrack = true,
                            .trackParticle = kPhoton1,
                            .trackPurity = 0.8f,
                            .leadingShare = 0.5};
    const auto out = classifyCandidate(g, dominance, config, in);
    CHECK(out.identity == kPi0);
    CHECK(out.calo);
    CHECK(out.merged);
    CHECK_FALSE(out.levelAssoc);
    CHECK(out.dominated);
    CHECK(out.purity == Catch::Approx(0.8));
    CHECK(out.trackCalo);
    CHECK(out.trackConsistent);
    CHECK_FALSE(out.pidCorrect);
    CHECK(out.identityBin == kIdentityMerged);
  }

  SECTION("a merged calorimeter identity with the track of another particle") {
    const CandidateInput in{.candidateClass = ParticleClass::ChargedHadron,
                            .hasTrackster = true,
                            .caloIdentity = kPi0,
                            .caloScore = 0.2f,
                            .hasTrack = true,
                            .trackParticle = kPion,
                            .leadingShare = 0.5};
    const auto out = classifyCandidate(g, dominance, config, in);
    CHECK(out.trackCalo);
    CHECK_FALSE(out.trackConsistent);
  }

  SECTION("a pileup identity") {
    const CandidateInput in{.candidateClass = ParticleClass::NeutralHadron,
                            .hasTrackster = true,
                            .caloIdentity = kOther,
                            .caloScore = 0.1f,
                            .leadingShare = -1.};
    const auto out = classifyCandidate(g, dominance, config, in);
    CHECK(out.associated);
    CHECK(out.pileup);
    CHECK_FALSE(out.levelAssoc);
    CHECK_FALSE(out.merged);
    CHECK(out.dominated);
    CHECK_FALSE(out.hasLevelCandidate);
    CHECK(out.identityBin == kIdentityPileup);
  }

  SECTION("a candidate below the share cut is not dominated") {
    const CandidateInput in{.candidateClass = ParticleClass::Photon,
                            .hasTrackster = true,
                            .caloIdentity = kPhoton1,
                            .caloScore = 0.4f,
                            .leadingShare = 0.4};
    const auto out = classifyCandidate(g, dominance, config, in);
    CHECK(out.levelAssoc);
    CHECK_FALSE(out.dominated);
    CHECK(out.pidCorrect);
    CHECK(out.purity == Catch::Approx(0.6));
  }
}
