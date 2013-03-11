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

  Vector center() const { return xp +Vector(m_alpha,m_beta)/m_c;}
  Vector momentum() const { return Vector(-m_beta,m_alpha)/std::abs(m_c);}
  Vector direction(Vector x0) const { return Vector(m_alpha,m_beta) - m_c *(x0-xp);}

  Scalar verify(Vector x) const {
    auto xx = x-xp;
    return m_c*xx.mag2() - 2*xx.dot(Vector(m_alpha,m_beta));
  }

private:
  Vector xp;
  T m_c;
  T m_alpha;
  T m_beta;
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
  auto seq = T(1.)+st2*st2;
  m_alpha  = std::copysign(T(1.)/std::sqrt(seq),det);
  m_c  = T(2.)*ct*m_alpha/det;  // now m_c has samesign of ct and alpha follows
  m_beta = -st2*m_alpha;
  if (!vert) std::swap(m_alpha,m_beta);
  xp = x2;

}

template<typename T> 
void TrajectoryCircle<T>::fromCurvTwoPoints(Scalar c, Vector x1, Vector x3) {
  auto xm = T(0.5)*(x1+x3);
  auto dir = x3-x1;
  auto d2 = dir.mag2();
  dir /= std::sqrt(d2); std::swap(dir[0],dir[1]); dir[1]=-dir[1];  // hope is correct,
  d2 *=T(0.25);
  auto sagitta = c*d2/(T(1)+std::sqrt(T(1)-c*c*d2));  // sign is correct?
  xp = xm-sagitta*dir;

  m_c = c;
  m_alpha = dir[0];
  m_beta = dir[1];
		       

}



#endif
