#ifndef TkSeedingLayers_SeedingLayer_H
#define TkSeedingLayers_SeedingLayer_H

#include <string>
#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
class DetLayer;
namespace edm { class Event; class EventSetup; };


namespace ctfseeding {

class SeedingLayer {
public:
  enum Side { Barrel = 0, NegEndcap =1,  PosEndcap = 2 }; 
public:
  SeedingLayer(const DetLayer* layer, const std::string & name, Side & side, int idLayer);
  ~SeedingLayer();

  std::string name() const { return theName; }
  std::vector<SeedingHit> hits(const edm::Event& ev, const edm::EventSetup& es) const;

  bool operator==(const SeedingLayer &s) const { return name()==s.name(); }

  const DetLayer*  detLayer() const { return theLayer; }
 
private:
  const DetLayer* theLayer;
  std::string theName;
  Side theSide;
  int theIdLayer;
};
}
#endif
