#ifndef LayerWithHits_H
#define LayerWithHits_H

class TrackingRecHit;
class DetLayer;

#include <vector>
#include <boost/iterator/indirect_iterator.hpp>
#include <algorithm>

class LayerWithHits
{
 public:
  LayerWithHits(const DetLayer *dl,std::vector<const TrackingRecHit*> theInputHits):
    theDetLayer(dl),theHits(theInputHits){}
  
  template<typename C> 
  LayerWithHits( const DetLayer *dl, typename C::Range range) {
    theDetLayer = dl;
    for(typename C::const_iterator id=range.first; id!=range.second; id++){
      size_t cs = theHits.size();
      theHits.resize(cs+range.second-range.first);
      std::copy((*id).begin(), (*id).end(),
		boost::make_indirect_iterator(theHits.begin()+cs));
    }
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

