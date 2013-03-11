#include<iostream>
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryCircle.h"
#include<algorithm>

template<typename T>
T go(T eps) {
  T merr = 0; T cerr=0; T span; T chord;

  using Vector = typename TrajectoryCircle<T>::Vector;
  using Circle = TrajectoryCircle<T>;

  T s2 = 1./std::sqrt(2);

  for (int iq=0; iq!=4; ++iq) {
    bool h = iq%2!=0; bool n = iq>1;
    Vector v[4] = { Vector(0.1,0.1), Vector(-0.4,1.1), Vector(0.1,2.1), Vector(0.6,1.1)};
    for (int i=0;i!=4;++i) {
      if(n) v[i][1]=-v[i][1]; if(h) std::swap(v[i][0],v[i][1]); 
    }
    for (int lr=0; lr!=3; ++lr) {
      if (lr==1) for (int i=0;i!=4;++i) { auto x= s2*(v[i][0]- v[i][1]); v[i][1]= s2*(v[i][0]+ v[i][1]);v[i][0]=x;}
      if (lr==2) for (int i=0;i!=4;++i) { auto x=  v[i][1]; v[i][1]= -v[i][0];v[i][0]=x;}
      std::cout << "cross " << (v[1]-v[0]).cross(v[2]-v[0]) << " " 
		<< (v[3]-v[0]).cross(v[2]-v[0]) << std::endl;
      auto vv = [&](Circle const & c1){
	auto cc1 = c1.center();
	auto p1 = c1.momentum();
	std::cout << c1.c0() << "  " << cc1[0] << "," << cc1[1] << "  " 
	<< p1[0] << "," << p1[1] << "   "
	<< c1.verify(v[0]) << " " << c1.verify(v[1]) 
	<< " " << c1.verify(v[2]) << " " << c1.verify(v[3]) << std::endl;
      };
      {
	Circle c1; c1.fromThreePoints(v[0],v[1],v[2]);
	vv(c1);
	Circle c2; c2.fromCurvTwoPoints(c1.c0(),v[0],v[2]);
	vv(c2);
	
      }
      {
	Circle c1; c1.fromThreePoints(v[0],v[3],v[2]);
	vv(c1);
	Circle c2; c2.fromCurvTwoPoints(c1.c0(),v[0],v[2]);
	vv(c2);
      }
    std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  T layers[] = {0.1f,5.f,8.f,10.f,20.f,30.f,50.f,70.f,80.f,100.f};

  int size = sizeof(layers)/sizeof(T);

  // full combinatorial...
  for (int q=0; q!=4; ++q) {  // quadrants
    std::cout << q << std::endl;
    bool h = q%2!=0; bool n = q>1;
    for (int i=0;i!=size-2;++i) {
      T x1= -2.*layers[i]; T d1 = -2*x1/20;
      for(;x1<2.*layers[i]; x1+=d1) {
	Vector xp1(x1,layers[i]); if(n) xp1[1]=-xp1[1]; if(h) std::swap(xp1[0],xp1[1]); 
	for (int j=i+1;j!=size-1;++j) {
	  T x2= -2.*layers[j]; T d2 = -2*x2/20;
	  for(;x2<2.*layers[j]; x2+=d2) {
	    Vector xp2(x2,layers[j]); if(n) xp2[1]=-xp2[1]; if(h) std::swap(xp2[0],xp2[1]);
	    auto doit = [&](int k, T x3, bool pr) {
	      Vector xp3(x3,layers[k]); if(n) xp3[1]=-xp3[1]; if(h) std::swap(xp3[0],xp3[1]); 
	      if ( (xp2-xp1).dot(xp3-xp2) > 0) {  // not if looping
		Circle circle; circle.fromThreePoints(xp1,xp2,xp3);
		T err = std::abs(circle.verify(xp1))+std::abs(circle.verify(xp2)) + std::abs(circle.verify(xp3));
		if (err>merr) { merr=err; cerr = circle.c0(); span=(xp3-xp1).mag(); chord = std::min((xp3-xp2).mag(),(xp2-xp1).mag());}
		Circle c2; c2.fromCurvTwoPoints(circle.c0(),xp1,xp2);
		auto cc2 = c2.center();
		T err2 = std::abs(c2.verify(xp1))+std::abs(c2.verify(xp2)) + std::abs(c2.verify(xp3));
		Circle c3; c3.fromCurvTwoPoints(circle.c0(),xp1,xp3);
		auto cc3 = c3.center();
		T err3 = std::abs(c3.verify(xp1))+std::abs(c3.verify(xp2)) + std::abs(c3.verify(xp3));
		if (pr || err > eps || err2 > 100.*eps  || err3 > 100.*eps) {
		  // std::cout << xp1 << " " << xp2 << " " << xp3 << std::endl;
		  std::cout << circle.c0() << " " << circle.verify(xp1) << " " << circle.verify(xp2) << " " << circle.verify(xp3) << std::endl;
		  std::cout << c2.c0() << " " << cc2[0] << "," << cc2[1] << "  "
			     << c2.verify(xp1) << " " << c2.verify(xp2) << " " << c2.verify(xp3) << std::endl;
		  std::cout << c3.c0()  << " "<< cc3[0] << "," << cc3[1] << "  "
			    << c3.verify(xp1) << " " << c3.verify(xp2) << " " << c3.verify(xp3) << std::endl;
		}
	      }
	    };
	    for (int k=j+1;k!=size;++k) {
	      T xl = x1 +  (x2-x1)*(layers[k]-layers[i])/(layers[j]-layers[i]);
	      doit(k,xl, false);doit(k,xl-0.001, false);doit(k,xl+0.001,false);
	      T x3= -2.*layers[k]; T d3 = -2*x3/20;
	      for(;x3<2.*layers[k]; x3+=d3) {
		doit(k, x3, false);
	      }
	    }
	  }
	}
      }
    }
  }
  
  std::cout << "max err " << merr << " for r0,span "  << 1./cerr <<  ", "  << span << ", "  << chord <<  std::endl;
  return merr;
  
}


int main() {

  std::cout << "float" << std::endl;
  go<float>(1.e-4);
  std::cout << "double" << std::endl;
  go<double>(1.e-5);


}
