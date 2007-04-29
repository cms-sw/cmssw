#ifndef MeasurementExtractor_H
#define MeasurementExtractor_H
 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

 
/** Extracts the subset of TrajectoryState parameters and errors
 *  that correspond to the parameters measured by a RecHit.
 */
 
class MeasurementExtractor {
 public:
  // construct
  MeasurementExtractor(const TrajectoryStateOnSurface& aTSoS) :
    theTSoS(aTSoS) {}
 
  // access
  
  // Following methods can be overloaded against their argument
  // thus allowing one to have different behaviour for different RecHit types
 
  AlgebraicVector measuredParameters(const  TransientTrackingRecHit&);
  AlgebraicSymMatrix measuredError(const  TransientTrackingRecHit&);

  template <unsigned int D> typename AlgebraicROOTObject<D>::Vector measuredParameters(const TransientTrackingRecHit &hit) {
      typedef typename AlgebraicROOTObject<D,5>::Matrix Mat;
      AlgebraicVector5 par5( theTSoS.localParameters().vector());
      Mat H = asSMatrix<D,5>( hit.projectionMatrix() );
      return H*par5;
  }

  template <unsigned int D> typename AlgebraicROOTObject<D>::SymMatrix measuredError(const TransientTrackingRecHit &hit) {
      typedef typename AlgebraicROOTObject<D,5>::Matrix Mat;
      const AlgebraicSymMatrix55 &err5 =  theTSoS.localError().matrix();
      Mat H = asSMatrix<D,5>( hit.projectionMatrix() );
      return ROOT::Math::Similarity(H,err5); 
  }

 private:
  const TrajectoryStateOnSurface& theTSoS;
};

#endif
 

