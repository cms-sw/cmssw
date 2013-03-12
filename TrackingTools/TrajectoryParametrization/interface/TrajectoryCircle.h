#ifndef TrajectoryCircle_H
#define TrajectoryCircle_H

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include <algorithm>

/*
 *  C*[(X-Xp)**2+(Y-Yp)**2] - 2*alpha*(X-Xp) - 2*beta*(Y-Yp) = 0             
 *    Xp,Yp is a point on the track     
 *    C = 1/r0 is the curvature  ( sign of C is charge of particle for B pos.) 
 *    alpha & beta are the direction cosines of the radial vector at Xp,Yp     
 *    i.e.  alpha = C*(X0-Xp),                                                 
 *          beta  = C*(Y0-Yp),                                                 
 *    where center of circle is at X0,Y0.                                      
 *    Slope dy/dx of tangent at Xp,Yp is -alpha/beta. 
 *   therefore px = -beta/abs(C) and py = alpha/abs(C)                          
 */

template<typename T> 
class  TrajectoryCircle {
public:
  using Scalar = T;
  using Vector = Basic2DVector<T>;
  using Double = double;

  inline void fromThreePoints(Vector x1, Vector x2, Vector x3);
  inline void fromCurvTwoPoints(Scalar c, Vector x1, Vector x3);

  Scalar c0() const { return m_c;}

  Vector center() const { return m_xp +m_dir/m_c;}
  Vector momentum() const { return Vector(-m_dir[1],m_dir[0])/std::abs(m_c);}
  Vector direction(Vector x0) const { return m_dir - m_c *(x0-m_xp);}

  Scalar verify(Vector x) const {
    auto xx = x-m_xp;
    return m_c*xx.mag2() - T(2)*xx.dot(m_dir);
  }


  inline Vector crossLine(Vector x0, Vector dir) const;

private:
  Vector m_xp;
  Vector m_dir; // alpha, beta
  T m_c;

};


template<typename T> 
void TrajectoryCircle<T>::fromThreePoints(Vector x1, Vector x2, Vector x3) {
  auto x1p = x1-x2;
  auto x3p = x3-x2;
  auto d12 = x1p.mag2();
  auto d32 = x3p.mag2();
  
  // the sign is the sign of curvature...
  auto ct  = x1p[0]*x3p[1]-x1p[1]*x3p[0];
  bool pos = ct > 0;
 

  // standard solution for vertical quadrant
  // swap if horizontal (really needed/correct???)
  bool vert = std::abs(x2[0]) < std::abs(x2[1]);
  int ix = vert ? 0 : 1;
  int iy = vert ? 1 : 0; 

  // bool down = (x1p[iy]>0); 
 
  // solution of linear equation... with sign following ct
  auto det = d12*x3p[iy]-d32*x1p[iy];
  auto st2 = (d12*x3p[ix]-d32*x1p[ix])/det;
  if (!vert) det  = -det;  // keep ct as is: so swap det
  //  Double seq = 1.+Double(st2)*Double(st2);
  auto seq = T(1.)+st2*st2;
  Scalar alpha  = std::copysign(T(1.)/std::sqrt(seq),det);
  m_c  = T(2.)*ct*alpha/det;  // now m_c has samesign of ct and alpha follows
  m_dir[0] = alpha;
  m_dir[1] = -st2*alpha;
  if (!vert) std::swap(m_dir[0],m_dir[1]);
  m_xp = x2;

}

template<typename T> 
void TrajectoryCircle<T>::fromCurvTwoPoints(Scalar c, Vector x1, Vector x3) {
  auto xm = T(0.5)*(x1+x3);
  auto dir = x3-x1;
  auto d2 = dir.mag2();
  dir /= std::sqrt(d2); std::swap(dir[0],dir[1]); dir[1]=-dir[1]; // toward center
  d2 *=T(0.25);
  auto sagitta = (c*d2)/(T(1)+std::sqrt(T(1.)-c*c*d2)); 
  m_xp = xm-sagitta*dir;
  m_dir = dir;
  m_c = c;
}

template<typename T> 
typename TrajectoryCircle<T>::Vector  TrajectoryCircle<T>::crossLine(Vector x0, Vector dir) const {
  // line parametrized as x = x0 + t*dir
  // we assume dir to be normalized (dir*dir=1)
  // return the solution closer to x0
  auto xx = x0-m_xp;
  auto a = m_c; // (*dir.mag2())
  auto b = m_c*dir.dot(xx) - m_dir.dot(dir);
  auto c = m_c*xx.mag2() - T(2)*m_dir.dot(xx);  // this is "verify"
  //Double det = (Double(b)*Double(b)-Double(a)*Double(c);
  auto det = b*b-a*c;
  Scalar q = b + std::copysign(std::sqrt(det),b);
  auto t = -c/q;   // -c/b even...
  return x0+t*dir;

}


#endif
