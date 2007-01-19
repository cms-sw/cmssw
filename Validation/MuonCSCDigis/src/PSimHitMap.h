#ifndef PSimHitMap_h
#define PSimHitMap_h

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <map>

class PSimHitMap
{
public:
  PSimHitMap(const std::string & collectionName);

  void fill(const edm::Event & e);

  const edm::PSimHitContainer & hits(int detId) const;

private:
  std::string theCollectionName;
  std::map<int, edm::PSimHitContainer> theMap;
  edm::PSimHitContainer theEmptyContainer;
};

#endif

