
#include "GEMCode/SimMuL1/interface/PSimHitMapCSC.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include <set>

namespace SimHitAnalysis {

//_____________________________________________________________________________
void 
PSimHitMapCSC::fill(const edm::Event & e)
{
  theMap.clear();
  theChLayers.clear();

  std::map<int, std::set<int> > mapSetChLayers;

  edm::Handle< edm::PSimHitContainer > hSimHits;
  e.getByLabel(theModuleName, theCollectionName, hSimHits);
  const edm::PSimHitContainer* simHits = hSimHits.product();
  for (edm::PSimHitContainer::const_iterator hit = simHits->begin();  hit != simHits->end();  ++hit)
  {
    unsigned layerid = hit->detUnitId();
    CSCDetId id(layerid);
    CSCDetId chid = id.chamberId();
    theMap[layerid].push_back(*hit);
    mapSetChLayers[chid.rawId()].insert(layerid);
  }

  for(std::map<int, std::set<int> >::const_iterator itr = mapSetChLayers.begin(); itr != mapSetChLayers.end(); ++itr)
  {
    std::vector<int> layers(itr->second.begin(),itr->second.end());
    theChLayers[itr->first] = layers;
  }

}


//_____________________________________________________________________________
std::vector<int> 
PSimHitMapCSC::chambersWithHits() const
{
  if (theChLayers.size()==0) return theEmptyVector;
  std::vector<int> result;
  result.reserve(theChLayers.size());
  for(std::map<int, std::vector<int> >::const_iterator itr = theChLayers.begin(); itr != theChLayers.end(); ++itr)
  {
    result.push_back(itr->first);
  }
  return result;
}


//_____________________________________________________________________________
std::vector<int> 
PSimHitMapCSC::chamberLayersWithHits(int detId) const
{
  std::map<int, std::vector<int> >::const_iterator itr = theChLayers.find(detId);
  if(itr != theChLayers.end()) return itr->second;
  else return theEmptyVector;
}

} // namespace SimHitAnalysis
