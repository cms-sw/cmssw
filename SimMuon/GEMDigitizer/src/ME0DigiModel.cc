#include "SimMuon/GEMDigitizer/interface/ME0DigiModel.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

void 
ME0DigiModel::fillDigis(int rollDetId, ME0DigiCollection& digis)
{
  for (auto d: strips_)
  {
    if (d.second == -999) continue;

    // (strip, bx)
    ME0Digi digi(d.first, d.second); 
    digis.insertDigi(ME0DetId(rollDetId), digi);
    addLinks(d.first, d.second);
  }
  strips_.clear();
}

void 
ME0DigiModel::addLinks(unsigned int strip, int bx)
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

  for(auto &charge: simTrackChargeMap)
  {
    const int simTrackId(charge.first);
    auto link(StripDigiSimLink(strip, simTrackId, eventIdMap[simTrackId], charge.second/totalCharge));
    stripDigiSimLinks_.push_back(link);
  }
}


