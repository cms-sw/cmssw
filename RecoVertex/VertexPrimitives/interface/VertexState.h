#ifndef VertexState_H
#define VertexState_H

#include "RecoVertex/VertexPrimitives/interface/BasicVertexState.h"
#include <vector>

/** Class containing a measurement of a vertex. Some data is calculated
 * on demand to improve performance.
 */

class VertexState : private  BasicVertexState::Proxy
{

  typedef BasicVertexState::Proxy             Base;

public:
  VertexState();
  VertexState(BasicVertexState* p);
  VertexState(const GlobalPoint & pos, const GlobalError & posErr,
  		const double & weightInMix = 1.0);
  VertexState(const GlobalPoint & pos, const GlobalWeight & posWeight,
  		const double & weightInMix = 1.0);
  VertexState(const AlgebraicVector & weightTimesPosition,
		const GlobalWeight & posWeight,
  		const double & weightInMix = 1.0);

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

  AlgebraicVector weightTimesPosition() const
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
  bool isValid() const {return Base::isValid();}

};

#endif

