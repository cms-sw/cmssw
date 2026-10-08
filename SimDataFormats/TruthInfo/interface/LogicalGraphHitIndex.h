// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef SimDataFormats_TruthInfo_LogicalGraphHitIndex_h
#define SimDataFormats_TruthInfo_LogicalGraphHitIndex_h

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace truth {

  // Detector channels of the hit index. Each channel keeps its own per-particle
  // direct hits and subgraph-aggregated hits, so calorimeter, tracker, MTD and
  // muon hits stay separate (different DetId spaces, metrics and recHit links).
  // Stored as the underlying type for dictionary simplicity, ordered inner ->
  // outer by detector radius. The value is the channel index and must stay stable;
  // code refers to channels by name, never by literal index.
  enum class HitChannel : uint8_t {
    Tracker = 0,  // tracker PSimHits, energy = energyLoss, no recHit link
    MTD = 1,      // MIP timing layer (BTL/ETL)
    Calo = 2,  // all calorimeter PCaloHits (HGCAL endcap + ECAL barrel + HCAL), recHit-mapped via the DetId->RecHit map
    Muon = 3   // muon chambers (DT/CSC/RPC/GEM)
  };
  inline constexpr std::size_t kNumHitChannels = 4;

  class LogicalGraphHitIndex {
  public:
    struct Hit {
      static constexpr uint32_t kInvalidRecHitIndex = std::numeric_limits<uint32_t>::max();

      uint32_t detId = 0;
      uint32_t recHitIndex = kInvalidRecHitIndex;
      float energy = 0.f;

      [[nodiscard]] bool hasRecHit() const { return recHitIndex != kInvalidRecHitIndex; }
    };

    // One detector channel. Two storage layouts exist, and which one an index carries
    // is a property of the data, not of the reading job: sharedSubgraphStore() reports
    // it and the accessors below handle both, so an index written either way reads back
    // correctly.
    //
    //   materialised: directOffsets/directHits hold each particle's own hits, and
    //       subgraphOffsets/subgraphHits hold a second, coalesced copy of every
    //       descendant's hits under each ancestor. A hit is stored once per ancestor
    //       that contains it. Written by every index that predates the shared layout.
    //
    //   shared (the default): dfsOffsets/directHits hold each hit exactly once, ordered
    //       so that a particle's descendants occupy the slots right after it. A subgraph
    //       is then a set of ranges of that single store and costs no hit storage at
    //       all. The hits of a range are in tree order rather than detId order and
    //       repeat a detId hit by several descendants, so a consumer that needs per-cell
    //       energies coalesces them.
    //
    // Members are public so the dictionary and the associator's flat scans can reach
    // them directly.
    struct Channel {
      std::vector<uint32_t> directOffsets;
      std::vector<Hit> directHits;
      std::vector<uint32_t> subgraphOffsets;
      std::vector<Hit> subgraphHits;
      std::vector<uint32_t> dfsOffsets;
    };

    // A run of consecutive DFS slots. Particles that carry hits own exactly one, their
    // own subtree. A GEN-only particle owns as many as it takes to cover the SIM
    // subtrees below it: the GEN half is a DAG, so its descendants are not one run.
    struct SlotRange {
      uint32_t firstSlot = 0;
      uint32_t slotCount = 0;
    };

    LogicalGraphHitIndex() = default;

    LogicalGraphHitIndex(uint32_t nParticles, std::vector<Channel> channels)
        : nParticles_(nParticles), channels_(std::move(channels)) {}

    LogicalGraphHitIndex(uint32_t nParticles,
                         std::vector<Channel> channels,
                         std::vector<uint32_t> dfsPos,
                         std::vector<uint32_t> rangeOffsets,
                         std::vector<SlotRange> ranges)
        : nParticles_(nParticles),
          channels_(std::move(channels)),
          dfsPos_(std::move(dfsPos)),
          rangeOffsets_(std::move(rangeOffsets)),
          ranges_(std::move(ranges)) {}

    // True when the shared layout is in use. The DFS slot of a particle and the ranges
    // its subgraph covers are properties of the tree, so they are stored once for the
    // whole index rather than per channel.
    [[nodiscard]] bool sharedSubgraphStore() const { return !dfsPos_.empty(); }
    [[nodiscard]] std::vector<uint32_t> const& dfsPos() const { return dfsPos_; }

    // The slot ranges a particle's subgraph covers. Empty in the materialised layout.
    [[nodiscard]] std::span<const SlotRange> subgraphRanges(uint32_t particleId) const {
      if (particleId + 1 >= rangeOffsets_.size())
        return {};
      const auto begin = rangeOffsets_[particleId];
      const auto end = rangeOffsets_[particleId + 1];
      return std::span<const SlotRange>(ranges_.data() + begin, end - begin);
    }

    // The hits of one slot range in a channel.
    [[nodiscard]] std::span<const Hit> rangeHits(HitChannel channel, SlotRange range) const {
      Channel const* channelData = channelOrNull(channel);
      if (channelData == nullptr || range.slotCount == 0)
        return {};
      if (range.firstSlot + range.slotCount >= channelData->dfsOffsets.size())
        return {};
      const auto begin = channelData->dfsOffsets[range.firstSlot];
      const auto end = channelData->dfsOffsets[range.firstSlot + range.slotCount];
      return std::span<const Hit>(channelData->directHits.data() + begin, end - begin);
    }

    [[nodiscard]] uint32_t nParticles() const { return nParticles_; }
    [[nodiscard]] static constexpr std::size_t nChannels() { return kNumHitChannels; }

    // Direct hits of a particle in a channel (the hits on its own SimTrack).
    [[nodiscard]] std::span<const Hit> directHits(HitChannel channel, uint32_t particleId) const {
      Channel const* channelData = channelOrNull(channel);
      if (channelData == nullptr)
        return {};
      if (sharedSubgraphStore()) {
        if (particleId >= dfsPos_.size())
          return {};
        return rangeHits(channel, SlotRange{dfsPos_[particleId], 1});
      }
      if (particleId + 1 >= channelData->directOffsets.size())
        return {};
      const auto begin = channelData->directOffsets[particleId];
      const auto end = channelData->directOffsets[particleId + 1];
      return std::span<const Hit>(channelData->directHits.data() + begin, end - begin);
    }

    // A particle's own hits plus those of every descendant, as one span. Coalesced and
    // sorted by detId in the materialised layout. In the shared layout this is valid
    // only for a particle whose subgraph is a single range, which is every particle
    // that carries hits; a GEN-only particle spans several ranges and returns empty
    // here, so a consumer that must handle those iterates subgraphRanges instead.
    [[nodiscard]] std::span<const Hit> subgraphHits(HitChannel channel, uint32_t particleId) const {
      Channel const* channelData = channelOrNull(channel);
      if (channelData == nullptr)
        return {};
      if (sharedSubgraphStore()) {
        const auto ranges = subgraphRanges(particleId);
        return ranges.size() == 1 ? rangeHits(channel, ranges[0]) : std::span<const Hit>{};
      }
      if (particleId + 1 >= channelData->subgraphOffsets.size())
        return {};
      const auto begin = channelData->subgraphOffsets[particleId];
      const auto end = channelData->subgraphOffsets[particleId + 1];
      return std::span<const Hit>(channelData->subgraphHits.data() + begin, end - begin);
    }

    // Append a particle's subgraph hits, whichever layout is in use. The only accessor
    // that is correct for every particle in both layouts.
    void appendSubgraphHits(HitChannel channel, uint32_t particleId, std::vector<Hit>& out) const {
      if (!sharedSubgraphStore()) {
        const auto span = subgraphHits(channel, particleId);
        out.insert(out.end(), span.begin(), span.end());
        return;
      }
      for (auto const& range : subgraphRanges(particleId)) {
        const auto span = rangeHits(channel, range);
        out.insert(out.end(), span.begin(), span.end());
      }
    }

    [[nodiscard]] bool hasChannel(HitChannel channel) const {
      Channel const* channelData = channelOrNull(channel);
      return channelData != nullptr && !channelData->directHits.empty();
    }

    // Raw channel storage (flat hit vectors + offsets), for callers that scan a
    // whole channel - e.g. BranchHitAssociator's inverted-index build.
    [[nodiscard]] Channel const& channel(HitChannel channel) const { return channels_.at(index(channel)); }

  private:
    [[nodiscard]] static constexpr std::size_t index(HitChannel channel) { return static_cast<std::size_t>(channel); }

    [[nodiscard]] Channel const* channelOrNull(HitChannel channel) const {
      const std::size_t channelIndex = index(channel);
      return channelIndex < channels_.size() ? &channels_[channelIndex] : nullptr;
    }

    uint32_t nParticles_ = 0;
    std::vector<Channel> channels_;  // size kNumHitChannels when built

    // Non-empty only in the shared layout: the DFS slot of each particle, and the slot
    // ranges of each particle's subgraph, CSR-style.
    std::vector<uint32_t> dfsPos_;
    std::vector<uint32_t> rangeOffsets_;
    std::vector<SlotRange> ranges_;
  };

}  // namespace truth

#endif
