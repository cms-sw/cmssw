#ifndef HelpertRecHit2DLocalPos_H
#define HelpertRecHit2DLocalPos_H

// TO BE FIXED: the name of this class should be changed in something more generic 
// since it is now used also for 1D Hits

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"


// make it dummy, alignment error added in BaseTrackerRecHit


class HelpertRecHit2DLocalPos //: public TValidTrackingRecHit 
{
public:

  static AlgebraicSymMatrix parError( const LocalError& le, const GeomDet& det) {
    AlgebraicSymMatrix m(2);
    m[0][0] = le.xx();
    m[0][1] = le.xy();
    m[1][1] = le.yy();
    return m;
  }

  /// Fills in KFComponents delegating to hit2dLocalPos, plus adding APE if available
  /// hit2dLocalPos MUST BE a 2D rechit measuring local position (e.g. BaseTrackerRecHit2D)
  static void getKfComponents( KfComponentsHolder & holder, 
			       const TrackingRecHit &hit2dLocalPos,
			       const GeomDet& det){hit2dLocalPos.getKfComponents(holder);}


  /// Fills in KFComponents delegating to hit1D, plus adding APE if available
  static void getKfComponents( KfComponentsHolder & holder, 
			       const SiStripRecHit1D& hit1D,
			       const GeomDet& det){hit1D.getKfComponents(holder);}

  static void updateWithAPE(LocalError& le, const GeomDet& det);
  
};

#endif
