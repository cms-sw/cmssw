#ifndef ThreePlaneCrossing_H
#define ThreePlaneCrossing_H

#include "DataFormats/GeometrySurface/interface/Plane.h"

class ThreePlaneCrossing {
public:


  Plane::GlobalPoint crossing( const Plane& a, const Plane& b, 
			       const Plane& c) const;

};

#endif
