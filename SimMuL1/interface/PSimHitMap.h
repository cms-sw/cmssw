#ifndef SimMuL1_PSimHitMap_h
#define SimMuL1_PSimHitMap_h

// Modified from the original 1_6_12 version of #include "SimMuon/MCTruth/interface/PSimHitMap.h"
// -- V. Khotilovich


#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <map>

namespace SimHitAnalysis {

class PSimHitMap
{
public:
  // defaults to "g4SimHits", "MuonCSCHits" and hits from PSimHitContainer
  PSimHitMap():
    useCrossingFrame(false),
    theModuleName("g4SimHits"),
    theCollectionName("MuonCSCHits"),
    theMap(),
    theEmptyContainer() {}

  // for filling from CrssingFrame only
  PSimHitMap(std::string & collectionName):
    useCrossingFrame(true),
    theModuleName(""),
    theCollectionName(collectionName),
    theMap(),
    theEmptyContainer() {}

  // for filling from PSimHitContainer only
  PSimHitMap(std::string & collectionName, std::string & moduleName):
    useCrossingFrame(false),
    theModuleName(moduleName),
    theCollectionName(collectionName),
    theMap(),
    theEmptyContainer() {}

  // customization 
  void setUseCrossingFrame(bool useCF) { useCrossingFrame = useCF;}
  void setCollectionName(std::string & collectionName) {theCollectionName = collectionName;}
  void setModuleName(std::string & moduleName) {theModuleName=moduleName;}
  void setInputTag(edm::InputTag &t);

  void fill(const edm::Event & e);

  const edm::PSimHitContainer & hits(int detId) const;

  std::vector<int> detsWithHits() const;
  std::map<int, edm::PSimHitContainer> getMap() const {return theMap;}

protected:
  bool useCrossingFrame;
  std::string theModuleName;
  std::string theCollectionName;
  std::map<int, edm::PSimHitContainer> theMap;
  edm::PSimHitContainer theEmptyContainer;
  std::vector<int> theEmptyVector;
};

}// namespace SimHitAnalysis

#endif
