#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/TrackingRecHit/interface/KfComponentsHolder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "DataFormats/Math/interface/ProjectMatrix.h"


// test of joseph form
#ifdef KU_JF_TEST

#include<atomic>
#include<iostream>
namespace {
  struct Stat {
    Stat(): tot(0),
    nopd(0), jnopd(0), fnopd(0),
    inopd(0), ijnopd(0), ifnopd(0),dtot(0)
    {for (int i=0;i<5;++i) idmm[i]=0; }

    std::atomic<long long> tot;
    std::atomic<long long> nopd;
    std::atomic<long long> jnopd;
    std::atomic<long long> fnopd;
    std::atomic<long long> inopd;
    std::atomic<long long> ijnopd;
    std::atomic<long long> ifnopd;

    std::atomic<unsigned long long> dtot;
    std::atomic<unsigned long long> idmm[5];
    double mm = 0;
    double dmm[5] = {0};
    bool ok=true;
    ~Stat() {
       double n = 1.e-3/double(dtot);
       std::cout << "KF " << tot*1.e-9 << "G " << mm <<" "
         <<dmm[0]<<'/'<<dmm[1]<<'/'<<dmm[2]<<'/'<<dmm[3]<<'/'<<dmm[4]<< '\n' 
         <<dtot << ' ' <<idmm[0]<<'/'<<idmm[1]<<'/'<<idmm[2]<<'/'<<idmm[3]<<'/'<<idmm[4]<< '\n'
         <<std::sqrt(idmm[0]*n)<<'/'<<std::sqrt(idmm[1]*n)<<'/'<<std::sqrt(idmm[2]*n)<<'/'<<std::sqrt(idmm[3]*n)<<'/'<<std::sqrt(idmm[4]*n)
         << " " << nopd << " " << jnopd << " " << fnopd
         << " " << inopd << " " << ijnopd << " " << ifnopd
         << std::endl;
     }
  };

  Stat stat;

  bool isNopd(AlgebraicSymMatrix55 const & m) {
    return m(0,0)<0 || m(1,1)<0 || m(2,2)<0 || m(3,3)<0 || m(4,4)<0;
  }
}
#endif


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

  auto && x = tsos.localParameters().vector();
  auto && C = tsos.localError().matrix();

  // projection matrix (assume element of "H" to be just 0 or 1)
  ProjectMatrix<double,5,D>  pf;
 
  // Measurement matrix
  VecD r, rMeas; 
  SMatDD V(SMatrixNoInit{}), VMeas(SMatrixNoInit{});

  KfComponentsHolder holder; 
  holder.template setup<D>(&r, &V, &pf, &rMeas, &VMeas, x, C);
  aRecHit.getKfComponents(holder);
  
  r -= rMeas;

  // and covariance matrix of residuals
  SMatDD R = V + VMeas;
  bool ok = invertPosDefMatrix(R);

  // Compute Kalman gain matrix
  AlgebraicMatrix55 M = AlgebraicMatrixID();
  Mat5D K = C*pf.project(R);
  pf.projectAndSubtractFrom(M,K);
 

  // Compute local filtered state vector
  AlgebraicVector5 fsv = x + K * r;

  // Compute covariance matrix of local filtered state vector 
#define KU_JosephForm
#ifdef KU_JosephForm
  AlgebraicSymMatrix55 fse = ROOT::Math::Similarity(M, C) + ROOT::Math::Similarity(K, V);
#else
  AlgebraicSymMatrix55 fse;  ROOT::Math::AssignSym::Evaluate(fse, M*C);
#endif

// test	of joseph form
#ifdef KU_JF_TEST

  AlgebraicSymMatrix55 fse2;  ROOT::Math::AssignSym::Evaluate(fse2, M*C);

  // std::cout << "Joseph Form \n" << fse << std::endl;
  // std::cout << "Fast Form \n"	<< fse2 << std::endl;


  stat.tot++;
  auto n1 = isNopd(fse);
  auto n2 = isNopd(fse2);
  if (n1&&n2) stat.nopd++;
  if (n1) stat.jnopd++;
  if (n2) stat.fnopd++;

  AlgebraicSymMatrix55 dd = fse2-fse;
  auto dda = dd.Array();
  auto fsa = fse.Array();
  double ddd[15];
  for (int i=0; i<15; ++i) ddd[i] = std::abs(dda[i])/std::abs(fsa[i]);
  auto mm = *std::max_element(ddd,ddd+15);
  stat.mm = std::max(stat.mm,mm);
  if (stat.ok && !(n1||n2)) {
   stat.dtot++;
   for (int i=0; i<5; ++i) {
     auto dmm = std::sqrt(fse2(i,i)) - std::sqrt(fse(i,i));
     stat.dmm[i] = std::max(stat.dmm[i],std::abs(dmm));
     stat.idmm[i] += (unsigned long long)(1000.*dmm*dmm);
     if (stat.idmm[i] > std::numeric_limits<long long>::max() ) stat.ok=false;
   }
  }

  AlgebraicSymMatrix55 ifse = fse; invertPosDefMatrix(ifse);
  AlgebraicSymMatrix55 ifse2 = fse2; invertPosDefMatrix(ifse2);
  n1 = isNopd(ifse);
  n2 = isNopd(ifse2);
  if (n1&&n2) stat.inopd++;
  if (n1) stat.ijnopd++;
  if (n2) stat.ifnopd++;
#endif

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

