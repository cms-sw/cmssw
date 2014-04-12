#ifndef GenericLinearizationPointFinder_H
#define GenericLinearizationPointFinder_H

#include "RecoVertex/VertexTools/interface/LinearizationPointFinder.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"

  /** A generic linearization point finder, that uses the result of 
   *  a Fitter to be used as a lin.point
   */

class GenericLinearizationPointFinder : public LinearizationPointFinder {

public:

  GenericLinearizationPointFinder ( const VertexFitter & fitter ) : 
    theFitter ( fitter.clone() ) {}

  ~GenericLinearizationPointFinder () { delete theFitter; }

  /** Method giving back the Initial Linearization Point.
   */
  virtual 
  GlobalPoint getLinearizationPoint(const std::vector<reco::TransientTrack> & tracks) const { 
    return theFitter->vertex ( tracks ).position(); 
  }

  /** Clone method
   */        
  virtual LinearizationPointFinder * clone() const {
    return new GenericLinearizationPointFinder(* this);
  }

private:

  const VertexFitter * theFitter;

};

#endif
