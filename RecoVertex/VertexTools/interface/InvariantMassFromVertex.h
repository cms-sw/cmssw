#ifndef InvariantMassFromVertex_H
#define InvariantMassFromVertex_H

// #include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
// #include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertexFactory.h"
// #include "RecoVertex/KinematicFitPrimitives/interface/VirtualKinematicParticleFactory.h"
#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
// #include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"
// #include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/Vector4D.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/**
 * Class building  a resulting output
 * tree out of the information provided
 * by KinematicParticleVertexFitter.
 */
class InvariantMassFromVertex{

public:
  typedef ROOT::Math::PxPyPzMVector LorentzVector;

  Measurement1D invariantMass(const CachingVertex<5>& vertex,
                          const std::vector<double> & masses) const;

  Measurement1D invariantMass(const CachingVertex<5>& vertex,
                          const double mass) const;

  /**
   * four-momentum Lorentz vector
   */
  LorentzVector p4 (const CachingVertex<5>& vertex,
                          const std::vector<double> & masses) const;

  /**
   * four-momentum Lorentz vector
   */
  LorentzVector p4 (const CachingVertex<5>& vertex,
                          const double mass) const;

  GlobalVector momentum(const CachingVertex<5>& vertex) const;


private:

 typedef ReferenceCountingPointer<VertexTrack<5> > RefCountedVertexTrack;
 typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;
 typedef ReferenceCountingPointer<RefittedTrackState<5> > RefCountedRefittedTrackState;

  double uncertainty(const LorentzVector & p4, const CachingVertex<5>& vertex,
	const std::vector<double> & masses) const;
};

#endif
