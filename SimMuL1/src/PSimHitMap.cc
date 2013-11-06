//
// Modified from the original 1_6_12 version of SimMuon/MCTruth/src/PSimHitMap.cc
// -- V. Khotilovich
//

#include "GEMCode/SimMuL1/interface/PSimHitMap.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

namespace SimHitAnalysis {

//_____________________________________________________________________________
void 
PSimHitMap::fill(const edm::Event & e)
{
  theMap.clear();
  theEmptyVector.clear();

  if (useCrossingFrame)
  {
// needs  update if ever it would be necessary to use simhits from CF
/*
    edm::Handle<CrossingFrame> cf;
    e.getByType(cf);

    MixCollection<PSimHit> simHits(cf.product(), theCollectionName);

    // arrange the hits by detUnit
    for(MixCollection<PSimHit>::MixItr hit = simHits.begin(); hit != simHits.end(); ++hit)
      theMap[hit->detUnitId()].push_back(*hit);
*/
  } 
  else
  {
    edm::Handle< edm::PSimHitContainer > hSimHits;
    e.getByLabel(theModuleName, theCollectionName, hSimHits);
    const edm::PSimHitContainer* simHits = hSimHits.product();
    for (edm::PSimHitContainer::const_iterator hit = simHits->begin();  hit != simHits->end();  ++hit) 
      theMap[hit->detUnitId()].push_back(*hit);
  }
}


//_____________________________________________________________________________
const edm::PSimHitContainer & 
PSimHitMap::hits(int detId) const
{
  std::map<int, edm::PSimHitContainer>::const_iterator mapItr = theMap.find(detId);
  if(mapItr != theMap.end())    return mapItr->second;
  else  return theEmptyContainer;
}


//_____________________________________________________________________________
std::vector<int> 
PSimHitMap::detsWithHits() const
{
  if (theMap.size()==0) return theEmptyVector;
  std::vector<int> result;
  result.reserve(theMap.size());
  for(std::map<int, edm::PSimHitContainer>::const_iterator mapItr = theMap.begin(); mapItr != theMap.end(); ++mapItr)
  {
    result.push_back(mapItr->first);
  }
  return result;
}


//_____________________________________________________________________________
void 
PSimHitMap::setInputTag(edm::InputTag &t)
{
  theModuleName = t.label();
  theCollectionName = t.instance();
}

} // namespace SimHitAnalysis
