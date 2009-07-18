#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

void
HelpertRecHit2DLocalPos::updateWithAPE(LocalError& le, const GeomDet& det) {
  if ( det.alignmentPositionError() != 0) {
    LocalError lape =
      ErrorFrameTransformer().transform( det.alignmentPositionError()->globalError(),
                                         det.surface());
    le = LocalError(le.xx()+lape.xx(),
                    le.xy()+lape.xy(),
                    le.yy()+lape.yy());
  }
}

AlgebraicSymMatrix HelpertRecHit2DLocalPos::parError( const LocalError& le,
						      const GeomDet& det)
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

void
HelpertRecHit2DLocalPos::getKfComponents(KfComponentsHolder & holder,
                        const TrackingRecHit &hit2dLocalPos,
                        const GeomDet& det)
{
    hit2dLocalPos.getKfComponents(holder);
    if ( det.alignmentPositionError() != 0) {
        LocalError lape =
            ErrorFrameTransformer().transform( det.alignmentPositionError()->globalError(),
                    det.surface());
        AlgebraicSymMatrix22 &m = holder.errors<2>();
        m(0, 0) += lape.xx();
        m(0, 1) += lape.xy();
        m(1, 1) += lape.yy();
    }
}
