#ifndef RefCountedKinematicParticle_H
#define RefCountedKinematicParticle_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"
//#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertex.h"

typedef ReferenceCountingPointer<KinematicParticle> RefCountedKinematicParticle;

//template<class T, class V>
//  bool operator==(const ReferenceCountingPointer<T> & rh, const
//   ReferenceCountingPointer<V> & lh)
//  { return rh.get() == lh.get();}

#endif
