#include "RecoVertex/VertexTools/interface/VertexCompatibleWithBeam.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include "RecoVertex/VertexTools/interface/VertexDistance.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/VertexTools/interface/BeamSpot.h"


VertexCompatibleWithBeam::VertexCompatibleWithBeam(const VertexDistance & d, 
						   float cut) 
  : theDistance(d.clone()), theCut(cut) 
{
  BeamSpot beamSpot;
  theBeam = VertexState(beamSpot.position(), beamSpot.error());
}


VertexCompatibleWithBeam::VertexCompatibleWithBeam(
  const VertexCompatibleWithBeam & other) : 
  theDistance((*other.theDistance).clone()), 
  theCut(other.theCut), theBeam(other.theBeam) {}


VertexCompatibleWithBeam::~VertexCompatibleWithBeam() {
  delete theDistance;
}


VertexCompatibleWithBeam & 
VertexCompatibleWithBeam::operator=(const VertexCompatibleWithBeam & other) 
{
  if (this == &other) return *this;

  theDistance = (*other.theDistance).clone();
  theCut = other.theCut;
  theBeam = other.theBeam;

  return *this;
}


bool VertexCompatibleWithBeam::operator()(const reco::Vertex & v) const 
{
  GlobalPoint p(Basic3DVector<float> (v.position()));
  VertexState vs(p, RecoVertex::convertError(v.covariance()));
  return (theDistance->distance(vs, theBeam).value() < theCut);
}


float VertexCompatibleWithBeam::distanceToBeam(const reco::Vertex & v) const
{
  GlobalPoint p(Basic3DVector<float> (v.position()));
  VertexState vs(p, RecoVertex::convertError(v.covariance()));
  return theDistance->distance(vs, theBeam).value();
}
