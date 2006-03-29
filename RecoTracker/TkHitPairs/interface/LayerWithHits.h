#ifndef LayerWithHits_H
#define LayerWithHits_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
class LayerWithHits
{

 public:

  //constructor for pixel rechit
  LayerWithHits( const DetLayer *dl,const SiPixelRecHitCollection::range ran): ddl(dl),PixRANGE(ran){
    _isPixel=true;
    _isStrip=false;
    _isStripMatched=false;};
  //constructor for strip rechit
  LayerWithHits( const DetLayer *dl,const SiStripRecHit2DLocalPosCollection::range ran): ddl(dl),StripRANGE(ran){
    _isPixel=false;
    _isStrip=true;
    _isStripMatched=false;   };
  //constructor for strip matched rechit
 LayerWithHits( const DetLayer *dl,const SiStripRecHit2DMatchedLocalPosCollection::range ran): ddl(dl),MatchedRANGE(ran){
    _isPixel=false;
    _isStrip=false;
    _isStripMatched=true;   };
  //destructor
  ~LayerWithHits(){};
  
  //detlayer
  const  DetLayer* layer()  const {return ddl;};
  //range
  SiPixelRecHitCollection::range PixRange()const {return PixRANGE;};
  SiStripRecHit2DLocalPosCollection::range  StripRange()const {return StripRANGE;};
  SiStripRecHit2DMatchedLocalPosCollection::range StripMatchedRange() const {return MatchedRANGE;};
  
  //type of trackingrechit
  bool isPixel() const {return _isPixel;};
  bool isStrip() const {return _isStrip;};
  bool isStripMatched() const {return _isStripMatched;};
 
 private:

  const DetLayer* ddl;
  const SiPixelRecHitCollection::range PixRANGE;
  const SiStripRecHit2DLocalPosCollection::range StripRANGE;
  const SiStripRecHit2DMatchedLocalPosCollection::range MatchedRANGE;
  
  bool _isPixel;
  bool _isStrip;
  bool _isStripMatched;
 };
#endif

