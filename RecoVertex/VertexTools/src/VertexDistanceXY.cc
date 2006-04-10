#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cfloat>


using namespace reco;

Measurement1D VertexDistanceXY::distance(const VertexState & vtx1, 
					 const VertexState & vtx2) const
{
  return distance(vtx1.position(), vtx1.error(),
  		  vtx2.position(), vtx2.error());
}


Measurement1D 
VertexDistanceXY::distance(const Vertex& vtx1, const Vertex & vtx2) const
{
  return distance(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		  RecoVertex::convertError(vtx1.error()),
  		  GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		  RecoVertex::convertError(vtx2.error()));
}


float VertexDistanceXY::compatibility(const VertexState & vtx1, 
				      const VertexState & vtx2) const
{
  return compatibility(vtx1.position(), vtx1.error(),
		       vtx2.position(), vtx2.error());
}


float 
VertexDistanceXY::compatibility(const Vertex& vtx1, const Vertex & vtx2) const
{
  return compatibility(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		       RecoVertex::convertError(vtx1.error()),
		       GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		       RecoVertex::convertError(vtx2.error()));
}


Measurement1D 
VertexDistanceXY::distance(const GlobalPoint & vtx1Position, 
			   const GlobalError & vtx1PositionError, 
			   const GlobalPoint & vtx2Position, 
			   const GlobalError & vtx2PositionError) const
{
    AlgebraicSymMatrix error = vtx1PositionError.matrix()
      + vtx2PositionError.matrix();
    GlobalVector diff = vtx1Position - vtx2Position;
    AlgebraicVector vDiff(3);
    vDiff[0] = diff.x();
    vDiff[1] = diff.y();
    vDiff[2] = 0.;
    
    double dist=sqrt(pow(diff.x(),2)+pow(diff.y(),2));
    
    double err2 = error.similarity(vDiff);
    double err  = 0;
    if( dist != 0) err = sqrt(err2)/dist;
       
    return Measurement1D(dist,err);
}


float 
VertexDistanceXY::compatibility(const GlobalPoint & vtx1Position, 
				const GlobalError & vtx1PositionError, 
				const GlobalPoint & vtx2Position, 
				const GlobalError & vtx2PositionError) const
{
  // error matrix of residuals
  AlgebraicSymMatrix err1 = vtx1PositionError.matrix();
  AlgebraicSymMatrix err2 = vtx2PositionError.matrix();
  AlgebraicSymMatrix error(2, 0); 
  error[0][0] = err1[0][0] + err2[0][0];
  error[0][1] = err1[0][1] + err2[0][1];
  error[1][1] = err1[1][1] + err2[1][1];
  if (error == theNullMatrix) return FLT_MAX;

  // position residuals
  GlobalVector diff = vtx2Position - vtx1Position;
  AlgebraicVector vDiff(2);
  vDiff[0] = diff.x();
  vDiff[1] = diff.y();

  // Invert error matrix of residuals
  int ifail;
  error.invert(ifail);
  if (ifail != 0) {
    throw cms::Exception("VertexDistanceXY::matrix inversion problem");
  }

  return error.similarity(vDiff);
}
