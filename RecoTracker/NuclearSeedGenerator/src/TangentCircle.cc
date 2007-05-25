#include "RecoTracker/NuclearSeedGenerator/interface/TangentCircle.h"

#define PI 3.1415926

TangentCircle::TangentCircle(const GlobalVector& direction, const GlobalPoint& inner, const GlobalPoint& outer) : 
       theInnerPoint(inner), theOuterPoint(outer), theVertexPoint(inner) {
   double x1 = inner.x();
   double y1 = inner.y();
   double x2 = outer.x();
   double y2 = outer.y();
   double alpha1 = (direction.y() != 0) ? atan(-direction.x()/direction.y()) : PI/2 ;
   double denominator = 2*((x1-x2)*cos(alpha1)+(y1-y2)*sin(alpha1));
   theRho = (denominator != 0) ? ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))/denominator : 1E12;

   // variable not calculated
   theX0 = 1000;
   theY0 = 1000;
}

TangentCircle::TangentCircle(const GlobalPoint& outerPoint, const GlobalPoint& innerPoint, const GlobalPoint& vertexPoint) : 
     theInnerPoint(theInnerPoint), theOuterPoint(outerPoint), theVertexPoint(vertexPoint) {
     FastCircle circle(outerPoint, innerPoint, vertexPoint);
     theX0 = circle.x0();
     theY0 = circle.y0();
     theRho = circle.rho();
}

GlobalVector TangentCircle::direction(const GlobalPoint& point) const {
     GlobalVector dir(point.y() - theY0, point.x() - theX0, point.z());
     dir/=dir.mag();
     return dir;
}
