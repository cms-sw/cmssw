#include "SimMuon/GEMDigitizer/interface/GEMDigiModel.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

void 
GEMDigiModel::fillDigis(int rollDetId, GEMDigiCollection& digis)
{
  for (const auto& d: strips_)
  {
    if (d.second == -999) continue;

    // (strip, bx)
    GEMDigi digi(d.first, d.second); 
    digis.insertDigi(GEMDetId(rollDetId), digi);
    addLinks(d.first, d.second);
    addLinksWithPartId(d.first, d.second);
  }
  strips_.clear();
}

void 
GEMDigiModel::addLinks(unsigned int strip, int bx)
{
  std::pair<unsigned int, int> digi(strip, bx);
  auto channelHitItr = detectorHitMap_.equal_range(digi);

  // find the fraction contribution for each SimTrack
  std::map<int, float> simTrackChargeMap;
  std::map<int, EncodedEventId> eventIdMap;
  float totalCharge(0.);
  for(auto hitItr = channelHitItr.first; hitItr != channelHitItr.second; ++hitItr)
  {
    const PSimHit * hit(hitItr->second);
    // might be zero for unit tests and such
    if(hit == nullptr) continue;
    
    int simTrackId(hit->trackId());
    //float charge = hit->getCharge();
    const float charge(1.f);
    auto chargeItr = simTrackChargeMap.find(simTrackId);
    if( chargeItr == simTrackChargeMap.end() )
    {
      simTrackChargeMap[simTrackId] = charge;
      eventIdMap[simTrackId] = hit->eventId();
    }
    else 
    {
      chargeItr->second += charge;
    }
    totalCharge += charge;
  }

  for (const auto& charge: simTrackChargeMap)
  {
    const int simTrackId(charge.first);
    auto link(StripDigiSimLink(strip, simTrackId, eventIdMap[simTrackId], charge.second/totalCharge));
    stripDigiSimLinks_.push_back(link);
  }
}

void GEMDigiModel::addLinksWithPartId(unsigned int strip, int bx)
{
 
  std::pair<unsigned int, int > digi(strip, bx);
  std::pair<DetectorHitMap::iterator, DetectorHitMap::iterator> channelHitItr 
     = detectorHitMap_.equal_range(digi);

  for( DetectorHitMap::iterator hitItr = channelHitItr.first; hitItr != channelHitItr.second; ++hitItr)
  {
    const PSimHit * hit = (hitItr->second);
    // might be zero for unit tests and such
    if (hit == nullptr) continue;

    theGemDigiSimLinks_.push_back(GEMDigiSimLink(digi, hit->entryPoint(), hit->momentumAtEntry(), hit->timeOfFlight(), hit->energyLoss(),
                                                       hit->particleType(), hit->detUnitId(), hit->trackId(), hit->eventId(), hit->processType()));

  }
}

