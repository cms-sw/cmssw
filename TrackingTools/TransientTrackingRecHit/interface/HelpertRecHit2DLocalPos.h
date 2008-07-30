#ifndef HelpertRecHit2DLocalPos_H
#define HelpertRecHit2DLocalPos_H

//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"

class LocalError;
class GeomDet;


class HelpertRecHit2DLocalPos //: public TransientTrackingRecHit 
{
public:

  AlgebraicSymMatrix parError( const LocalError& le, const GeomDet& det) const;

  /// Fills in KFComponents delegating to hit2dLocalPos, plus adding APE if available
  /// hit2dLocalPos MUST BE a 2D rechit measuring local position (e.g. BaseSiTrackerRecHit2DLocalPos)
  void getKfComponents( KfComponentsHolder & holder, 
                        const TrackingRecHit &hit2dLocalPos,
                        const GeomDet& det) const ;
};

#endif
