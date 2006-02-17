#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"


const GeomDetUnit * TransientTrackingRecHit::detUnit() const 
{
//   if(theDet_) return(theDet_) ;
//   if(_detMap[_id]) return(_detMap[_id]);
//   setDet() ;
//   return _detMap[_id] ;
return _geom->idToDet(geographicalId());
}
