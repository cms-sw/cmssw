#ifndef LayerWithHits_H
#define LayerWithHits_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetAlgorithm.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

class LayerWithHits
{
 public:
  LayerWithHits(const DetLayer *dl,const std::vector<const TrackingRecHit*>& theInputHits):
    theDetLayer(dl),theHits(theInputHits){}
  
  /// Usage: 
  ///  TrackerLayerIdAccessor acc;
  ///  LayerWithHits( theLayer, collrphi, acc.stripTIBLayer(1) );
  template <typename DSTV, typename SEL>
  LayerWithHits(const DetLayer *dl,
                DSTV const & allhits,    
                SEL  const & sel) 
  {
    theDetLayer = dl;
    edmNew::copyDetSetRange(allhits,theHits,sel);
  } 


  //destructor
  ~LayerWithHits(){}
  
  /// return the recHits of the Layer
  const std::vector<const TrackingRecHit*>& recHits() const {return theHits;}

  //detlayer
  const  DetLayer* layer()  const {return theDetLayer;}
  
 private:
  const DetLayer* theDetLayer;
  std::vector<const TrackingRecHit*> theHits;
};
#endif

