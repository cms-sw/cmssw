#ifndef LayerWithHits_H
#define LayerWithHits_H

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"
class LayerWithHits
{
 public:



  LayerWithHits( const PixelBarrelLayer *dl,const SiPixelRecHitCollection::range ran): ddl(dl),RANGE(ran){
    /*     SiPixelRecHitCollection::const_iterator k; */
    /*     for (k=ran.first;k!=ran.second;k++){ */
    /*       cout<<&(*k)<<" kkkk"<<endl;   */
    /*   }     */
    //    cout<<&(*(ran.first))<<" pippo  "<<&(*(ran.second))<<endl;
    //   abort();
  };



  ~LayerWithHits(){};

  const  PixelBarrelLayer* layer()  const {//cout<< "test radius in Layer(): " << ddl->specificSurface().radius() << endl;
return ddl;};
  SiPixelRecHitCollection::range Range()const {return RANGE;};

 private:
 const  PixelBarrelLayer* ddl;

  const    SiPixelRecHitCollection::range RANGE;
};
#endif

