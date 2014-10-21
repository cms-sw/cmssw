#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

void PSimHitMap::fill(const edm::Event & e)
{
  theMap.clear();
  edm::Handle<CrossingFrame<PSimHit> > cf;
  LogTrace("MuonCSCDigis") << "getting CrossingFrame<PSimHit> collection ";
  e.getByToken(sh_token, cf);

  MixCollection<PSimHit> simHits(cf.product());
  LogTrace("MuonCSCDigis") <<"... size = "<<simHits.size();

  // arrange the hits by detUnit                                                                          
  for(MixCollection<PSimHit>::MixItr hitItr = simHits.begin();
      hitItr != simHits.end(); ++hitItr)
    {
      theMap[hitItr->detUnitId()].push_back(*hitItr);
    }
}

const edm::PSimHitContainer & PSimHitMap::hits(int detId) const
{
  std::map<int, edm::PSimHitContainer>::const_iterator mapItr
    = theMap.find(detId);
  if(mapItr != theMap.end())
    {
      return mapItr->second;
    }
  else
    {
      return theEmptyContainer;
    }
}


std::vector<int> PSimHitMap::detsWithHits() const 
{
  std::vector<int> result;
  result.reserve(theMap.size());
  for(std::map<int, edm::PSimHitContainer>::const_iterator mapItr = theMap.begin(),
	mapEnd = theMap.end();
      mapItr != mapEnd;
      ++mapItr)
    {
      result.push_back(mapItr->first);
    }
  return result;
} 
