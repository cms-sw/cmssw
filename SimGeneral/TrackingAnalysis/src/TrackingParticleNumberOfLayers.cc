#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleNumberOfLayers.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

namespace {
  bool trackIdHitPairLess(const std::pair<unsigned int, const PSimHit*>& a, const std::pair<unsigned int, const PSimHit*>& b) {
    return a.first < b.first;
  }
}

TrackingParticleNumberOfLayers::TrackingParticleNumberOfLayers(const edm::Event& iEvent, const std::vector<edm::EDGetTokenT<std::vector<PSimHit> > >& simHitTokens) {

  // A multimap linking SimTrack::trackId() to a pointer to PSimHit
  // Similar to TrackingTruthAccumulator
  for(const auto& simHitToken: simHitTokens) {
    edm::Handle<std::vector<PSimHit> > hsimhits;
    iEvent.getByToken(simHitToken, hsimhits);
    trackIdToHitPtr_.reserve(trackIdToHitPtr_.size()+hsimhits->size());
    for(const auto& simHit: *hsimhits) {
      trackIdToHitPtr_.emplace_back(simHit.trackId(), &simHit);
    }
  }
  std::stable_sort(trackIdToHitPtr_.begin(), trackIdToHitPtr_.end(), trackIdHitPairLess);
}

std::tuple<std::unique_ptr<edm::ValueMap<unsigned int>>,
           std::unique_ptr<edm::ValueMap<unsigned int>>,
           std::unique_ptr<edm::ValueMap<unsigned int>>>
TrackingParticleNumberOfLayers::calculate(const edm::Handle<TrackingParticleCollection>& htps, const edm::EventSetup& iSetup) const {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology& tTopo = *tTopoHandle;

  const TrackingParticleCollection& tps = *htps;
  std::vector<unsigned int> valuesLayers(tps.size(), 0);
  std::vector<unsigned int> valuesPixelLayers(tps.size(), 0);
  std::vector<unsigned int> valuesStripMonoAndStereoLayers(tps.size(), 0);
  for(size_t iTP=0; iTP<tps.size(); ++iTP) {
    const TrackingParticle& tp = tps[iTP];
    const auto pdgId = tp.pdgId();

    // I would prefer a better way...
    constexpr unsigned int maxSubdet = static_cast<unsigned>(StripSubdetector::TEC)+1;
    constexpr unsigned int maxLayer = 0xF+1; // as in HitPattern.h
    bool hasHit[maxSubdet][maxLayer];
    bool hasPixel[maxSubdet][maxLayer];
    bool hasMono[maxSubdet][maxLayer];
    bool hasStereo[maxSubdet][maxLayer];
    memset(hasHit, 0, sizeof(hasHit));
    memset(hasPixel, 0, sizeof(hasPixel));
    memset(hasMono, 0, sizeof(hasMono));
    memset(hasStereo, 0, sizeof(hasStereo));

    for(const SimTrack& simTrack: tp.g4Tracks()) {
      // Logic is from TrackingTruthAccumulator
      auto range = std::equal_range(trackIdToHitPtr_.begin(), trackIdToHitPtr_.end(), std::pair<unsigned int, const PSimHit *>(simTrack.trackId(), nullptr), trackIdHitPairLess);
      if(range.first == range.second) continue;

      auto iHitPtr = range.first;
      int processType = iHitPtr->second->processType();
      int particleType = iHitPtr->second->particleType();

      for(; iHitPtr != range.second; ++iHitPtr) {
        const PSimHit& simHit = *(iHitPtr->second);
        if(simHit.eventId() != tp.eventId())
          continue;
        DetId newDetector = DetId( simHit.detUnitId() );

        // Check for delta and interaction products discards
        if( processType == simHit.processType() && particleType == simHit.particleType() && pdgId == simHit.particleType() ) {
          // The logic of this piece follows HitPattern
          bool isPixel = false;
          bool isStripStereo = false;
          bool isStripMono = false;

          switch(newDetector.subdetId()) {
          case PixelSubdetector::PixelBarrel:
          case PixelSubdetector::PixelEndcap:
            isPixel = true;
            break;
          case StripSubdetector::TIB:
            isStripMono = tTopo.tibIsRPhi(newDetector);
            isStripStereo = tTopo.tibIsStereo(newDetector);
            break;
          case StripSubdetector::TID:
            isStripMono = tTopo.tidIsRPhi(newDetector);
            isStripStereo = tTopo.tidIsStereo(newDetector);
            break;
          case StripSubdetector::TOB:
            isStripMono = tTopo.tobIsRPhi(newDetector);
            isStripStereo = tTopo.tobIsStereo(newDetector);
            break;
          case StripSubdetector::TEC:
            isStripMono = tTopo.tecIsRPhi(newDetector);
            isStripStereo = tTopo.tecIsStereo(newDetector);
            break;
          }

          const auto subdet = newDetector.subdetId();
          const auto layer = tTopo.layer( newDetector );

          hasHit[subdet][layer] = true;
          if(isPixel)            hasPixel[subdet][layer]  = isPixel;
          else if(isStripMono)   hasMono[subdet][layer]   = isStripMono;
          else if(isStripStereo) hasStereo[subdet][layer] = isStripStereo;
        }
      }
    }


    unsigned int nLayers = 0;
    unsigned int nPixelLayers = 0;
    unsigned int nStripMonoAndStereoLayers = 0;
    for(unsigned int i=0; i<maxSubdet; ++i) {
      for(unsigned int j=0; j<maxLayer; ++j) {
        nLayers += hasHit[i][j];
        nPixelLayers += hasPixel[i][j];
        nStripMonoAndStereoLayers += (hasMono[i][j] && hasStereo[i][j]);
      }
    }

    valuesLayers[iTP] = nLayers;
    valuesPixelLayers[iTP] = nPixelLayers;
    valuesStripMonoAndStereoLayers[iTP] = nStripMonoAndStereoLayers;
  }

  auto ret0 = std::make_unique<edm::ValueMap<unsigned int>>();
  {
    edm::ValueMap<unsigned int>::Filler filler(*ret0);
    filler.insert(htps, valuesLayers.begin(), valuesLayers.end());
    filler.fill();
  }
  auto ret1 = std::make_unique<edm::ValueMap<unsigned int>>();
  {
    edm::ValueMap<unsigned int>::Filler filler(*ret1);
    filler.insert(htps, valuesPixelLayers.begin(), valuesPixelLayers.end());
    filler.fill();
  }
  auto ret2 = std::make_unique<edm::ValueMap<unsigned int>>();
  {
    edm::ValueMap<unsigned int>::Filler filler(*ret2);
    filler.insert(htps, valuesStripMonoAndStereoLayers.begin(), valuesStripMonoAndStereoLayers.end());
    filler.fill();
  }

  return std::make_tuple(std::move(ret0), std::move(ret1), std::move(ret2));
}
