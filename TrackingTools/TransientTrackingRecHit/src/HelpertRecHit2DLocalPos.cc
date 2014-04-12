#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"


void
HelpertRecHit2DLocalPos::updateWithAPE(LocalError& le, const GeomDet& det) 
{
  LocalError lape = det.localAlignmentError();
  if (lape.valid())
    le = LocalError(le.xx()+lape.xx(),
		    le.xy()+lape.xy(),
		    le.yy()+lape.yy()
		    );
}

/*
AlgebraicSymMatrix HelpertRecHit2DLocalPos::parError( const LocalError& le,
						      const GeomDet& det)
{
  AlgebraicSymMatrix m(2);
  LocalError lape = det.localAlignmentError();
  if (lape.valid()) {
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
  LocalError lape = det.localAlignmentError();
  if (lape.valid()) {
    AlgebraicSymMatrix22 &m = holder.errors<2>();
    m(0, 0) += lape.xx();
    m(0, 1) += lape.xy();
    m(1, 1) += lape.yy();
  }
}

void
HelpertRecHit2DLocalPos::getKfComponents(KfComponentsHolder & holder,
					 const SiStripRecHit1D& hit1D,
					 const GeomDet& det)
{
  hit1D.getKfComponents(holder);
  LocalError lape =det.localAlignmentError();
  if (lape.valid()) {
    AlgebraicSymMatrix11 &m = holder.errors<1>();
    m(0, 0) += lape.xx();
  }
}
*/
