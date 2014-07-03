#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"

namespace {

template <unsigned int D>
TrajectoryStateOnSurface lupdate(const TrajectoryStateOnSurface& tsos,
				           const TrackingRecHit& aRecHit) {

  typedef typename AlgebraicROOTObject<D,5>::Matrix MatD5;
  typedef typename AlgebraicROOTObject<5,D>::Matrix Mat5D;
  typedef typename AlgebraicROOTObject<D,D>::SymMatrix SMatDD;
  typedef typename AlgebraicROOTObject<D>::Vector VecD;
  using ROOT::Math::SMatrixNoInit;
  double pzSign = tsos.localParameters().pzSign();

  //MeasurementExtractor me(tsos);

  auto && x = tsos.localParameters().vector();
  auto && C = tsos.localError().matrix();

  // projection matrix (should be zeroed)
  ProjectMatrix<double,5,D>  pf;
  MatD5 H;
 
  // Measurement matrix
  VecD r, rMeas; 
  SMatDD V(SMatrixNoInit{}), VMeas(SMatrixNoInit{});

  KfComponentsHolder holder; 
  holder.template setup<D>(&r, &V, &H, &pf, &rMeas, &VMeas, x, C);
  aRecHit.getKfComponents(holder);
  
  r -= rMeas;

  // and covariance matrix of residuals
  SMatDD R = V + VMeas;
  bool ok = invertPosDefMatrix(R);

  // Compute Kalman gain matrix
  Mat5D K(SMatrixNoInit{});
  AlgebraicMatrix55 M = AlgebraicMatrixID();
  if likely(holder.useProjFunc() ) {
    K = C*pf.project(R);
    pf.projectAndSubtractFrom(M,K);
  }
  else {
    K = (C * ROOT::Math::Transpose(H)) * R;
    M -=  K * H;
  }

  // Compute local filtered state vector
  AlgebraicVector5 fsv = x + K * r;
  // Compute covariance matrix of local filtered state vector
  AlgebraicSymMatrix55 fse = ROOT::Math::Similarity(M, C) + ROOT::Math::Similarity(K, V);


  /*
  // expanded similariy
  AlgebraicSymMatrix55 fse; 
  ROOT::Math::AssignSym::Evaluate(fse, (M* C) * ( ROOT::Math::Transpose(M)));
  AlgebraicSymMatrix55 tmp;
  ROOT::Math::AssignSym::Evaluate(tmp, (K*V) * (ROOT::Math::Transpose(K)));
  fse +=  tmp;
  */

  if (ok) {
    return TrajectoryStateOnSurface( LocalTrajectoryParameters(fsv, pzSign),
				     LocalTrajectoryError(fse), tsos.surface(),&(tsos.globalParameters().magneticField()), tsos.surfaceSide() );
  }else {
    edm::LogError("KFUpdator")<<" could not invert martix:\n"<< (V+VMeas);
    return TrajectoryStateOnSurface();
  }

}
}

TrajectoryStateOnSurface KFUpdator::update(const TrajectoryStateOnSurface& tsos,
                                           const TrackingRecHit& aRecHit) const {
    switch (aRecHit.dimension()) {
        case 1: return lupdate<1>(tsos,aRecHit);
        case 2: return lupdate<2>(tsos,aRecHit);
        case 3: return lupdate<3>(tsos,aRecHit);
        case 4: return lupdate<4>(tsos,aRecHit);
        case 5: return lupdate<5>(tsos,aRecHit);
    }
    throw cms::Exception("Rec hit of invalid dimension (not 1,2,3,4,5)") <<
         "The value was " << aRecHit.dimension() <<
        ", type is " << typeid(aRecHit).name() << "\n";
}

