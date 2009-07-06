#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"


#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"


#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>
#include<vector>

bool isAligned(void* data, long alignment)
{
 // check that the alignment is a power of two
 assert((alignment & (alignment-1)) == 0); 
 return ((long)data & (alignment-1)) == 0;
}


void st(){}
void en(){}


typedef DistanceBetweenComponents<5> Distance; 
typedef KullbackLeiblerDistance<5> KDistance;
typedef SingleGaussianState<5> GS;
typedef GS::Vector Vector;
typedef GS::Matrix Matrix;
typedef ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> > Matrix6;

  Distance const & distance() {
      static Distance * d = new KDistance;
      return *d;
  }

Matrix buildCovariance(float y) {

  // build a resonable covariance matrix as JIJ

  Basic3DVector<float>  axis(0.5,1.,1);
  
  Surface::RotationType rot(axis,0.5*M_PI);

  Surface::PositionType pos( 0., 0., 0.);

  Plane plane(pos,rot);
  LocalTrajectoryParameters tp(1., 1., y, 0.,0.,1.);

  JacobianLocalToCartesian jl2c(plane,tp);
  return ROOT::Math::SimilarityT(jl2c.jacobian(),Matrix6(ROOT::Math::SMatrixIdentity()));
  // return  ROOT::Math::Transpose(jl2c.jacobian())* jl2c.jacobian();

}

int main(int argc, char * argv[]) {


  std::cout << "size of  SingleGaussianState<5>" << sizeof(GS)
	    << std::endl;
  std::cout << "size of  Matrix" << sizeof(Matrix)
	    << std::endl;
  std::cout << "size of  Vector" << sizeof(Vector)
	    << std::endl;


  Distance const & d = distance();

  Matrix cov1 = buildCovariance(1.);
  Matrix cov2 = buildCovariance(2.);

  GS * gs1 = new GS(Vector(1., 1.,1., 1.,1.),cov1);
  // GS gs1(Vector(1., 1.,1., 1.,1.),Matrix(ROOT::Math::SMatrixIdentity()));



  GS * gs0 = new GS(Vector(1., 1.,1., 0.,0.),Matrix(ROOT::Math::SMatrixIdentity()));
  GS * gsP = new GS(Vector(1., 1.,1., 10.,10.),Matrix(ROOT::Math::SMatrixIdentity()));

 std::cout << "GS " << ((isAligned(gs0,16)) ? "a " : "n ") << std::endl;	
 std::cout << "cov " << ((isAligned(&gs0->covariance(),16)) ? "a " : "n ") << std::endl;	
 std::cout << "mean " << (((&gs0->mean(),16)) ? "a " : "n ") << std::endl;	
 std::cout << "weightM " << (((&gs0->weightMatrix(),16)) ? "a " : "n ") << std::endl;	



  // GS gs2(Vector(2., 2., 2., 2.,2.),cov);
  GS * gs2 = new GS(Vector(2., 2., 2., 2.,2.),cov2);
 
  std::vector<GS> vgs(10000);
  vgs.front() = *gs1;
  vgs.back() = *gs2;


  // make sure we load all code...
  edm::HRTimeType s0= edm::hrRealTime();
  double res = d(*gs0,*gsP);
  edm::HRTimeType e0 = edm::hrRealTime();
  std::cout << e0-s0 << std::endl;

  double res2=0;

  if (argc<2) return 1;

  if (argv[1][0]=='a') { 
    st();	
    edm::HRTimeType s= edm::hrRealTime();
    res2 = d(*gs1,*gs2);
    edm::HRTimeType e = edm::hrRealTime();
    en();
    std::cout << e-s << std::endl;
  } 
  else if (argv[1][0]=='b') { 
    st();	
    edm::HRTimeType s= edm::hrRealTime();
    res2 = d(vgs.front(),vgs.back());
    edm::HRTimeType e = edm::hrRealTime();
    en();
    std::cout << e-s << std::endl;
  } 

  std:: cout << res << " " << res2 << std::endl;

  return 0;

}
