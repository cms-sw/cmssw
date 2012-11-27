#ifndef HelpertRecHit2DLocalPos_H
#define HelpertRecHit2DLocalPos_H

// TO BE FIXED: the name of this class should be changed in something more generic 
// since it is now used also for 1D Hits

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"

class LocalError;
class GeomDet;


class HelpertRecHit2DLocalPos //: public TValidTrackingRecHit 
{
public:

  static AlgebraicSymMatrix parError( const LocalError& le, const GeomDet& det);

  /// Fills in KFComponents delegating to hit2dLocalPos, plus adding APE if available
  /// hit2dLocalPos MUST BE a 2D rechit measuring local position (e.g. BaseSiTrackerRecHit2DLocalPos)
  static void getKfComponents( KfComponentsHolder & holder, 
			       const TrackingRecHit &hit2dLocalPos,
			       const GeomDet& det);


  /// Fills in KFComponents delegating to hit1D, plus adding APE if available
  static void getKfComponents( KfComponentsHolder & holder, 
			       const SiStripRecHit1D& hit1D,
			       const GeomDet& det);

  static void updateWithAPE(LocalError& le, const GeomDet& det) ;
  
};

#endif
