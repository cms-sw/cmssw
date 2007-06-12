#include "RecoVertex/VertexPrimitives/interface/BeamSpot.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"

BeamSpot::BeamSpot() : thePos(0, 0, 0), 
		       theErr(0.0015*0.0015, 0., 0.0015*0.0015, 
			      0., 0., 5.3*5.3) 
{}

BeamSpot::BeamSpot(const reco::BeamSpot & beamSpot) :
	 thePos(RecoVertex::convertPos(beamSpot.position()))
{
  GlobalVector newZ(beamSpot.dxdz(), beamSpot.dydz(), 1);
  AlgebraicSymMatrix diagError(3,0);
  diagError(1,1) = pow(beamSpot.BeamWidth(),2) + pow(beamSpot.x0Error(),2);
  diagError(2,2) = pow(beamSpot.BeamWidth(),2) + pow(beamSpot.y0Error(),2);
  diagError(3,3) = pow(beamSpot.sigmaZ(),2) + pow(beamSpot.z0Error(),2);
  theErr = rotateMatrix(newZ, diagError);
}

BeamSpot::BeamSpot(const BeamSpotObjects * beamSpot)
{
  thePos=GlobalPoint(beamSpot->GetX(), beamSpot->GetY(), beamSpot->GetZ());

  GlobalVector newZ(beamSpot->Getdxdz(), beamSpot->Getdydz(), 1);
  AlgebraicSymMatrix diagError(3,0);
  diagError(1,1) = pow(beamSpot->GetBeamWidth(),2)+pow(beamSpot->GetXError(),2);
  diagError(2,2) = pow(beamSpot->GetBeamWidth(),2)+pow(beamSpot->GetYError(),2);
  diagError(3,3) = pow(beamSpot->GetSigmaZ(),2) +  pow(beamSpot->GetZError(),2);
  theErr = rotateMatrix(newZ, diagError);
}

AlgebraicSymMatrix BeamSpot::rotateMatrix(const GlobalVector& newZ,
	const AlgebraicSymMatrix& diagError) const
{
  GlobalVector globalZ(0.,0.,1.);
  GlobalVector rotationAxis = globalZ.unit().cross(newZ.unit());
  float rotationAngle = -acos(globalZ.unit().dot(newZ.unit()));

  TkRotation<float> rotation(rotationAxis.basicVector(),rotationAngle);
  AlgebraicMatrix rotationMatrix(3,3,0);
  rotationMatrix(1,1) = rotation.xx();
  rotationMatrix(1,2) = rotation.xy();
  rotationMatrix(1,3) = rotation.xz();
  rotationMatrix(2,1) = rotation.yx();
  rotationMatrix(2,2) = rotation.yy();
  rotationMatrix(2,3) = rotation.yz();
  rotationMatrix(3,1) = rotation.zx();
  rotationMatrix(3,2) = rotation.zy();
  rotationMatrix(3,3) = rotation.zz();

//  std::cout << rotation;

  return diagError.similarity(rotationMatrix);

}
