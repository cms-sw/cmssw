#include <cmath>
#include <cstdio>
#include <iostream>


inline double fastPow(double a, double b) {
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;
  return u.d;
}



int main() {
  constexpr double Safety = 0.9;
  double eps=1.e-5;
  const float cut = std::pow(10.f/Safety,5.f);
  double accMin = eps/cut;
  std::cout << "eps/cut/accMin " << eps << " " << cut << " " << accMin
	    << std::endl;
  for (double acc=accMin; acc<eps;  acc*=10.)
    std::cout << acc << " " << std::pow(eps/acc,0.2f) << " " << fastPow(eps/acc,0.2f) << std::endl;

  return 0;


}
