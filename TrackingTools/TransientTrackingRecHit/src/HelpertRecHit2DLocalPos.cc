#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

AlgebraicSymMatrix HelpertRecHit2DLocalPos::parError( const LocalError& le,
						      const GeomDet& det) const
{
  AlgebraicSymMatrix m(2);
  if ( det.alignmentPositionError() != 0) {
    LocalError lape =
      ErrorFrameTransformer().transform( det.alignmentPositionError()->globalError(),
                                         det.surface());
    m[0][0] = le.xx()+lape.xx();
    m[0][1] = le.xy()+lape.xy();
    m[1][1] = le.yy()+lape.yy();
  } else {
    m[0][0] = le.xx();
    m[0][1] = le.xy();
    m[1][1] = le.yy();
  };
  return m;
}
