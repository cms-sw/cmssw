#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include <cfloat>


using namespace reco;

Measurement1D
VertexDistance3D::signedDistance(const Vertex& vtx1, const Vertex & vtx2,
					 const GlobalVector & momentum) const
{
  Measurement1D unsignedDistance = distance(vtx1, vtx2);
  Basic3DVector<float> diff = Basic3DVector<float> (vtx2.position()) - 
    Basic3DVector<float> (vtx1.position());
//   Basic3DVector<float> (vtx2 - vtx1);
  if ((momentum.x()*diff.x() + momentum.y()*diff.y() * momentum.z()*diff.z()) < 0 )
    return Measurement1D(-1.0*unsignedDistance.value(),unsignedDistance.error());
  return unsignedDistance;
}


Measurement1D 
VertexDistance3D::distance(const GlobalPoint & vtx1Position, 
			   const GlobalError & vtx1PositionError, 
			   const GlobalPoint & vtx2Position, 
			   const GlobalError & vtx2PositionError) const
{
    AlgebraicSymMatrix33 error = vtx1PositionError.matrix()
      + vtx2PositionError.matrix();
    GlobalVector diff = vtx1Position - vtx2Position;
    AlgebraicVector3 vDiff;
    vDiff[0] = diff.x();
    vDiff[1] = diff.y();
    vDiff[2] = diff.z();
    
    double dist=diff.mag();
    
    double err2 = ROOT::Math::Similarity(error,vDiff);
    double err = 0.;
    if (dist != 0) err  =  sqrt(err2)/dist;
 
     return Measurement1D(dist,err);
}



float 
VertexDistance3D::compatibility(const GlobalPoint & vtx1Position, 
				const GlobalError & vtx1PositionError, 
				const GlobalPoint & vtx2Position, 
				const GlobalError & vtx2PositionError) const
{
  // error matrix of residuals
  AlgebraicSymMatrix33 err1 = vtx1PositionError.matrix();
  AlgebraicSymMatrix33 err2 = vtx2PositionError.matrix();
  AlgebraicSymMatrix33 error = err1 + err2;
  if (error == theNullMatrix) return FLT_MAX;

  // position residuals
  GlobalVector diff = vtx2Position - vtx1Position;
  AlgebraicVector3 vDiff;
  vDiff[0] = diff.x();
  vDiff[1] = diff.y();
  vDiff[2] = diff.z();

  // Invert error matrix of residuals
  bool ifail = !error.InvertChol();
  if (ifail) {
    throw cms::Exception("VertexDistance3D::matrix inversion problem");
  }

  return ROOT::Math::Similarity(error,vDiff);
}
