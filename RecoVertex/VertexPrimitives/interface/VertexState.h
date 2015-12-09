#ifndef VertexState_H
#define VertexState_H

#include "RecoVertex/VertexPrimitives/interface/BasicVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/BasicSingleVertexState.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include <vector>

/** Class containing a measurement of a vertex. Some data is calculated
 * on demand to improve performance.
 */

class VertexState final : private  BasicVertexState::Proxy {

  using Base =  BasicVertexState::Proxy;
  using BSVS =  BasicSingleVertexState;
public:
  VertexState(){}
  VertexState(VertexState const&) = default; 
  VertexState(VertexState &&) = default;
  VertexState & operator=(const VertexState&) = default;
  VertexState & operator=(VertexState&&) = default;

  // template<typename... Args>
  //  VertexState(Args && ...args) :
  //  Base ( new BSVS ( std::forward<Args>(args)...)){}

  explicit VertexState(BasicVertexState* p) : 
    Base(p) {}
  
  explicit VertexState(const reco::BeamSpot& beamSpot) :
    Base ( new BSVS ( GlobalPoint(Basic3DVector<float> (beamSpot.position())), 
		      GlobalError(beamSpot.rotatedCovariance3D()), 1.0)) {}
  

  VertexState(const GlobalPoint & pos, 
	      const GlobalError & posErr, const double & weightInMix= 1.0) :
    Base ( new BSVS (pos, posErr, weightInMix)) {}
  
  VertexState(const GlobalPoint & pos, 
	      const GlobalWeight & posWeight, const double & weightInMix= 1.0) :
    Base ( new BSVS (pos, posWeight, weightInMix)) {}
  
  VertexState(const AlgebraicVector3 & weightTimesPosition,
	      const GlobalWeight & posWeight, const double & weightInMix= 1.0) :
    Base ( new BSVS (weightTimesPosition, posWeight, weightInMix)) {}
  


  GlobalPoint position() const
  {
    return data().position();
  }

  GlobalError error() const
  {
    return data().error();
  }

  GlobalWeight weight() const
  {
    return data().weight();
  }

  AlgebraicVector3 weightTimesPosition() const
  {
    return data().weightTimesPosition();
  }

  double weightInMixture() const
  {
    return data().weightInMixture();
  }

  /** conversion to VertexSeed
   */
//   RefCountedVertexSeed seedWithoutTracks() const
//   {
//     return data().seedWithoutTracks();
//   }

  std::vector<VertexState> components() const
  {
    return data().components();
  }

  /// Make the ReferenceCountingProxy method to check validity public
  bool isValid() const {return Base::isValid() && data().isValid();}

};

#endif

