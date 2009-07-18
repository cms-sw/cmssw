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

  static AlgebraicSymMatrix parError( const LocalError& le, const GeomDet& det);

  /// Fills in KFComponents delegating to hit2dLocalPos, plus adding APE if available
  /// hit2dLocalPos MUST BE a 2D rechit measuring local position (e.g. BaseSiTrackerRecHit2DLocalPos)
  static void getKfComponents( KfComponentsHolder & holder, 
                        const TrackingRecHit &hit2dLocalPos,
                        const GeomDet& det);

  static void updateWithAPE(LocalError& le, const GeomDet& det) ;
  
};

#endif
