#include "RecoVertex/VertexPrimitives/interface/BasicSingleVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

typedef BasicSingleVertexState              BSVS;

VertexState::VertexState():
  Base ( new BSVS ()) {}

VertexState::VertexState(BasicVertexState* p) : 
  Base(p) {}

VertexState::VertexState(const GlobalPoint & pos, 
                         const GlobalError & posErr, 
                         const double & weightInMix) :
  Base ( new BSVS (pos, posErr, weightInMix)) {}

VertexState::VertexState(const GlobalPoint & pos, 
                         const GlobalWeight & posWeight,	
                         const double & weightInMix) :
  Base ( new BSVS (pos, posWeight, weightInMix)) {}

VertexState::VertexState(const AlgebraicVector3 & weightTimesPosition,
                         const GlobalWeight & posWeight,	
                         const double & weightInMix) :
  Base ( new BSVS (weightTimesPosition, posWeight, weightInMix)) {}


VertexState::VertexState(const GlobalPoint & pos, 
                         const double time,
                         const GlobalError & posTimeErr, 
                         const double & weightInMix) :
  Base ( new BSVS (pos, time, posTimeErr, weightInMix)) {}

VertexState::VertexState(const GlobalPoint & pos, 
                         const double time,
                         const GlobalWeight & posTimeWeight,	
                         const double & weightInMix) :
  Base ( new BSVS (pos, time, posTimeWeight, weightInMix)) {}

VertexState::VertexState(const AlgebraicVector4 & weightTimesPosition,
                         const GlobalWeight & posTimeWeight,	
                         const double & weightInMix) :
  Base ( new BSVS (weightTimesPosition, posTimeWeight, weightInMix)) {}

VertexState::VertexState(const reco::BeamSpot& beamSpot) :
  Base ( new BSVS ( GlobalPoint(Basic3DVector<float> (beamSpot.position())), 
  	GlobalError(beamSpot.rotatedCovariance3D()), 1.0)) {}
