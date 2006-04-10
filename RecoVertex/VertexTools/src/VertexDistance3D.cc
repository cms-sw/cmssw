#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cfloat>


using namespace reco;

Measurement1D VertexDistance3D::distance(const VertexState & vtx1, 
					 const VertexState & vtx2) const
{
  return distance(vtx1.position(), vtx1.error(),
  		  vtx2.position(), vtx2.error());
}


Measurement1D 
VertexDistance3D::distance(const Vertex & vtx1, const Vertex & vtx2) const
{
  return distance(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		  RecoVertex::convertError(vtx1.error()),
  		  GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		  RecoVertex::convertError(vtx2.error()));
}


Measurement1D 
VertexDistance3D::distance(const GlobalPoint & vtx1Position, 
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
    vDiff[2] = diff.z();
    
    double dist=diff.mag();
    
    double err2 = error.similarity(vDiff);
    double err = 0.;
    if (dist != 0) err  =  sqrt(err2)/dist;
 
     return Measurement1D(dist,err);
}


float VertexDistance3D::compatibility(const VertexState & vtx1, 
				      const VertexState & vtx2) const
{
  return compatibility(vtx1.position(), vtx1.error(),
		       vtx2.position(), vtx2.error());
}


float VertexDistance3D::compatibility(const Vertex & vtx1, 
				      const Vertex & vtx2) const
{
  return compatibility(GlobalPoint(Basic3DVector<float> (vtx1.position())), 
		       RecoVertex::convertError(vtx1.error()),
		       GlobalPoint(Basic3DVector<float> (vtx2.position())), 
		       RecoVertex::convertError(vtx2.error()));
}


float 
VertexDistance3D::compatibility(const GlobalPoint & vtx1Position, 
				const GlobalError & vtx1PositionError, 
				const GlobalPoint & vtx2Position, 
				const GlobalError & vtx2PositionError) const
{
  // error matrix of residuals
  AlgebraicSymMatrix err1 = vtx1PositionError.matrix();
  AlgebraicSymMatrix err2 = vtx2PositionError.matrix();
  AlgebraicSymMatrix error = err1 + err2;
  if (error == theNullMatrix) return FLT_MAX;

  // position residuals
  GlobalVector diff = vtx2Position - vtx1Position;
  AlgebraicVector vDiff(3);
  vDiff[0] = diff.x();
  vDiff[1] = diff.y();
  vDiff[2] = diff.z();

  // Invert error matrix of residuals
  int ifail;
  error.invert(ifail);
  if (ifail != 0) {
    throw cms::Exception("VertexDistance3D::matrix inversion problem");
  }

  return error.similarity(vDiff);
}
