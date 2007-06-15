#include "RecoTracker/NuclearSeedGenerator/interface/TangentHelix.h"

TangentHelix::TangentHelix(const TangentHelix& primCircle, const GlobalPoint& outerPoint, const GlobalPoint& innerPoint) :
     theInnerPoint(innerPoint), theOuterPoint(outerPoint), theCircle(primCircle.circle(), outerPoint, innerPoint) {

   // Calculation of vertex.z :
   GlobalPoint inner_T( innerPoint.x() , innerPoint.y() , 0.0 );
   GlobalPoint outer_T( outerPoint.x() , outerPoint.y() , 0.0 );
   GlobalPoint vtx_T( theCircle.vertexPoint().x() , theCircle.vertexPoint().y() , 0.0 );

   double d1 = (inner_T - vtx_T).mag();
   double d = (inner_T - outer_T).mag();

   theVertexPoint = GlobalPoint(vtx_T.x() , vtx_T.y(), innerPoint.z() - (outerPoint.z() - innerPoint.z()) * d1 / d );
   theDirectionAtVertex = GlobalVector(1000, 1000, 1000);
}

GlobalVector TangentHelix::directionAtVertex() {

   if(theDirectionAtVertex.z() > 999) {
      GlobalPoint inner_T( theInnerPoint.x() , theInnerPoint.y() , 0.0 );
      GlobalPoint outer_T( theOuterPoint.x() , theOuterPoint.y() , 0.0 );
      double p_z = (theOuterPoint.z() - theInnerPoint.z()) / (outer_T - inner_T).mag();

      GlobalVector dir_T = theCircle.directionAtVertex();
      GlobalVector dir( dir_T.x(), dir_T.y(), p_z);

      dir/=dir.mag();
      theDirectionAtVertex = dir;
    }

   return theDirectionAtVertex;
}
