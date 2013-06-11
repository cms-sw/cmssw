#ifndef SimMuL1_PSimHitMapCSC_h
#define SimMuL1_PSimHitMapCSC_h

#include "PSimHitMap.h"

namespace SimHitAnalysis {

class PSimHitMapCSC: public PSimHitMap
{
public:
  void fill(const edm::Event & e);

  std::vector<int> chambersWithHits() const;
  std::vector<int> chamberLayersWithHits(int detId) const;

private:
  std::map<int, std::vector<int> > theChLayers;
};

} // namespace SimHitAnalysis

#endif
