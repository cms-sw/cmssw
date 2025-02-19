#include "RecoVertex/VertexPrimitives/interface/BasicSingleVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

typedef BasicSingleVertexState              BSVS;

VertexState::VertexState():
  Base ( new BSVS ()) {}

VertexState::VertexState(BasicVertexState* p) : 
  Base(p) {}

VertexState::VertexState(const GlobalPoint & pos, 
	const GlobalError & posErr, const double & weightInMix) :
  Base ( new BSVS (pos, posErr, weightInMix)) {}

VertexState::VertexState(const GlobalPoint & pos, 
	const GlobalWeight & posWeight,	const double & weightInMix) :
  Base ( new BSVS (pos, posWeight, weightInMix)) {}

VertexState::VertexState(const AlgebraicVector3 & weightTimesPosition,
	const GlobalWeight & posWeight,	const double & weightInMix) :
  Base ( new BSVS (weightTimesPosition, posWeight, weightInMix)) {}

VertexState::VertexState(const reco::BeamSpot& beamSpot) :
  Base ( new BSVS ( GlobalPoint(Basic3DVector<float> (beamSpot.position())), 
  	GlobalError(beamSpot.rotatedCovariance3D()), 1.0)) {}
