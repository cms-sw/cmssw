#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CAPipelineCounters_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CAPipelineCounters_h

// Pipeline stage counters for the CA tracking diagnostic funnel. Defined outside
// ALPAKA_ACCELERATOR_NAMESPACE so the enum values are usable from any translation unit.
namespace caHitNtupletGenerator {

  // Doublet rejection cut indices (for kDblRejBase offset access)
  enum DblRejCut : int {
    kCutInvalidHit = 0,
    kCutInnerCoord = 1,
    kCutClusterCut = 2,
    kCutInvalidModule = 3,
    kCutOuterCoord = 4,
    kCutDzRange = 5,
    kCutZ0 = 6,
    kCutPhi = 7,
    kCutZSize = 8,
    kCutPt = 9,
    kCutStubSigma = 10,
    kCutPixStub = 11,
    kNDblCuts = 12
  };

  // Number of doublet rejection cuts, used in the kDblRejBase offset arithmetic.
  constexpr int kNCuts = int(DblRejCut::kNDblCuts);

  // Group offsets: group index * kNCuts added to kDblRejBase
  enum DblRejGroup : int {
    kGroupTotal = 0,    // all pairs
    kGroupOTEarly = 1,  // inner layer 28 or 29
    kGroupOTLate = 2,   // inner layer 30, 31, or 32
    kNGroups = 3
  };

  // Triplet rejection cut indices (for kTrpRejBase offset access)
  enum TrpRejCut : int {
    kCutAlignedRZ = 0,
    kCutAlignedXY,
    kCutBeamspotCompatibleXY,
    kCutPhiCompatible,
    kCutSameSignDPhi,
    kCutStubsCurvCompatibleWithTriplet,
    kCutStubsCompatibleWithInnerDoublet,
    kNTrpCuts
  };

  enum PipelineCounter : int {
    kDoubletsTotal = 0,
    kDoubletsPixPix,
    kDoubletsPixOT,
    kDoubletsOTOT,
    // Per-layer-pair doublet counts: OT barrel consecutive (5)
    kDoubletsL28L29,
    kDoubletsL29L30,
    kDoubletsL30L31,
    kDoubletsL31L32,
    kDoubletsL32L33,
    // Flat/tilted module type breakdown for barrel-barrel doublets (15)
    kDoubletsL28L29_FF,
    kDoubletsL28L29_FT,
    kDoubletsL28L29_TT,
    kDoubletsL29L30_FF,
    kDoubletsL29L30_FT,
    kDoubletsL29L30_TT,
    kDoubletsL30L31_FF,
    kDoubletsL30L31_FT,
    kDoubletsL30L31_TT,
    kDoubletsL31L32_FF,
    kDoubletsL31L32_FT,
    kDoubletsL31L32_TT,
    kDoubletsL32L33_FF,
    kDoubletsL32L33_FT,
    kDoubletsL32L33_TT,
    // OT barrel layer to the first disk at z < 0 (CA 44 PS or 45 2S), by barrel layer (6)
    kDoubletsL28D1B,
    kDoubletsL29D1B,
    kDoubletsL30D1B,
    kDoubletsL31D1B,
    kDoubletsL32D1B,
    kDoubletsL33D1B,
    // OT barrel layer to the first disk at z > 0 (CA 34 PS or 35 2S), by barrel layer (6)
    kDoubletsL28D1F,
    kDoubletsL29D1F,
    kDoubletsL30D1F,
    kDoubletsL31D1F,
    kDoubletsL32D1F,
    kDoubletsL33D1F,
    // Disk-to-next-disk chain at z < 0 (CA 44-53, PS or 2S layers of consecutive disks) (4)
    kDoubletsD1BD2B,
    kDoubletsD2BD3B,
    kDoubletsD3BD4B,
    kDoubletsD4BD5B,
    // Disk-to-next-disk chain at z > 0 (CA 34-43) (4)
    kDoubletsD1FD2F,
    kDoubletsD2FD3F,
    kDoubletsD3FD4F,
    kDoubletsD4FD5F,
    kDoubletsOTOther,  // fallback for unexpected OT-OT pairs
    kTripletsTotal,
    kTripletsPixPixPix,
    kTripletsPixPixOT,
    kTripletsPixOTOT,
    kTripletsOTOTOT,
    // OOO triplet region breakdown (6)
    kTripletsOOO_barrel,       // all 3 layers in OT barrel (28-33)
    kTripletsOOO_brlToBwd,     // starts barrel, ends in the z < 0 endcap (44-53)
    kTripletsOOO_brlToFwd,     // starts barrel, ends in the z > 0 endcap (34-43)
    kTripletsOOO_bwd,          // all 3 layers in the z < 0 endcap (44-53)
    kTripletsOOO_fwd,          // all 3 layers in the z > 0 endcap (34-43)
    kTripletsOOO_other,        // any other OOO combination
    kTripletPhiMiddleRej,      // triplets rejected by phi residual at middle hit
    kTripletChainPhiResidRej,  // connections rejected early by chainPhiResidCut (before cellNeighbors)
    kFishboneKilled,
    kNtupletsTotal,
    kNtupletsWithOT,
    kNtupletsOT3Plus,
    // Final track quality distribution (after all dup removal and classification)
    kQualTotal,         // total tracks with hits > 0
    kQualBad,           // quality = bad (NaN fit, doublets, or unclassified)
    kQualEdup,          // quality = edup (early duplicate removed)
    kQualDup,           // quality = dup (fast duplicate removed)
    kQualLoose,         // quality = loose
    kQualStrict,        // quality >= strict
    kQualStrictWithOT,  // quality >= strict AND has OT hits
    kQualTight,         // quality >= tight
    kQualTightWithOT,   // quality >= tight AND has OT hits
    kQualHP,            // quality = highPurity
    kQualHPWithOT,      // highPurity tracks with at least 1 OT hit
    // Per-nhits quality breakdown
    // Tracks with 3-4 hits (triplets/quadruplets)
    kQualStrict34,  // strict with 3-4 hits
    kQualTight34,   // tight with 3-4 hits
    kQualHP34,      // HP with 3-4 hits
    // Tracks with 5 hits (quintuplets)
    kQualStrict5,  // strict with 5 hits
    kQualTight5,   // tight with 5 hits
    kQualHP5,      // HP with 5 hits
    // Tracks with 6+ hits
    kQualStrict6p,  // strict with 6+ hits
    kQualTight6p,   // tight with 6+ hits
    kQualHP6p,      // HP with 6+ hits
    // Chi2 boundary tracks (within 10% of threshold, at risk of quality flip)
    kChi2Boundary34,  // 3-4 hits with 0.9 <= chi2 < 1.1 (threshold=1.0)
    kChi2Boundary5,   // 5 hits with 2.7 <= chi2 < 3.3 (threshold=3.0)
    kChi2Boundary6p,  // 6+ hits with 4.5 <= chi2 < 5.5 (threshold=5.0)
    // Fishbone hits on tracks
    kTracksFishbone0,   // tracks with 0 fishbone hits
    kTracksFishbone1,   // tracks with 1 fishbone hit
    kTracksFishbone2p,  // tracks with 2+ fishbone hits
    // Cell status after the fishbone kill phase
    kCellsUsedInTriplet,  // cells with kUsed status bit set (participated in >=1 triplet)
    kCellsKilledTotal,    // cells with isKilled() after fishbone
    kCellsAlive,          // cells NOT killed (alive at find_ntuplets entry)
    // Cell-track associations from find_ntuplets
    kCellTrackPairs,  // total cell->track associations (= *nCellTracks, sizes deviceTracksCells_)
    // Per-cut doublet rejection: 3 groups (Total, OTEarly L28-29, OTLate L30-32) x 12 cuts.
    // Cut indices 0-11: invalidHit, innerCoord, clusterCut, invalidModule, outerCoord,
    //                 dzRange, z0Cut, phiCut, zSizeCut, ptCut, stubSigma, pixStub.
    kDblRejBase,  // base for computed-offset access: counter = kDblRejBase + group*12 + cut
    // base for computed-offset access: counter = kTrpRejBase + cut
    kTrpRejBase = kDblRejBase + int(DblRejCut::kNDblCuts) * int(DblRejGroup::kNGroups),  // 3 groups x 12 cuts
    kTotalCounters = kTrpRejBase + int(TrpRejCut::kNTrpCuts)                             // 7 cuts
  };

}  // namespace caHitNtupletGenerator

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CAPipelineCounters_h
