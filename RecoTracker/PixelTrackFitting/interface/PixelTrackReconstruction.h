#ifndef RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstruction_H
#define RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstruction_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

#include <memory>

class PixelFitter;
class PixelTrackFilter;
class RegionsSeedingHitSets;

class PixelTrackReconstruction {
public:
  PixelTrackReconstruction(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);
  ~PixelTrackReconstruction();

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  void run(pixeltrackfitting::TracksWithTTRHs& tah, edm::Event& ev, const edm::EventSetup& es);

private:
  edm::EDGetTokenT<RegionsSeedingHitSets> theHitSetsToken;
  edm::EDGetTokenT<PixelFitter> theFitterToken;
  edm::EDGetTokenT<PixelTrackFilter> theFilterToken;
  edm::ESGetToken<PixelTrackCleaner, PixelTrackCleaner::Record> theCleanerToken;
};
#endif
