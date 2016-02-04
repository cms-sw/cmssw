#ifndef SeedLayerPairs_H
#define SeedLayerPairs_H

/** \class SeedLayerPairs
 * abstract interface to acces pairs of layers 
 */

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
class DetLayer;
class LayerWithHits;

class SeedLayerPairs {
public:

  typedef std::pair< const LayerWithHits*, const LayerWithHits*>        LayerPair;

  SeedLayerPairs() {};
  virtual ~SeedLayerPairs() {};
    virtual std::vector<LayerPair> operator()()= 0;
  

};

#endif
