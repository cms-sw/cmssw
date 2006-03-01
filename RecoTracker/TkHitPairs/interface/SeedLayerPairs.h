#ifndef SeedLayerPairs_H
#define SeedLayerPairs_H

/** \class SeedLayerPairs
 * abstract interface to acces pairs of layers 
 */

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
class DetLayer;
class LayerWithHits;
using namespace std;
class SeedLayerPairs {
public:

  typedef pair< const LayerWithHits*, const LayerWithHits*>        LayerPair;

  virtual ~SeedLayerPairs() {}

  virtual vector<LayerPair> operator()() const = 0;

};

#endif
