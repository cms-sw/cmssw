#ifndef HelpertRecHit2DLocalPos_H
#define HelpertRecHit2DLocalPos_H

//#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

class LocalError;
class GeomDet;


class HelpertRecHit2DLocalPos //: public TransientTrackingRecHit 
{
public:

  AlgebraicSymMatrix parError( const LocalError& le, const GeomDet& det) const;

};

#endif
