#ifndef INC_ANGLESUTIL
#define INC_ANGLESUTIL
///////////////////////////////////////////////////////////////////////////////
// File: AnglesUtil.hpp
// 
// Purpose:  Provide useful functions for calculating angles and eta
//
// Created:   4-NOV-1998  Serban Protopopescu
// History:   replaced old KinemUtil  
// Modified:  23-July-2000 Add rapidity calculation
//            14-Aug-2003 converted all functions to double precision
///////////////////////////////////////////////////////////////////////////////
// Dependencies (#includes)

#include <cmath>
#include <cstdlib>


namespace kinem{
  const double PI=2.0*acos(0.);
  const double TWOPI=2.0*PI;
  const float ETA_LIMIT=15.0;
  const float EPSILON=1.E-10;

  //  calculate phi from x, y
  inline double phi(double x, double y);
  //  calculate phi for a line defined by xy1 and xy2 (xy2-xy1)
  inline double phi(double xy1[2], double xy2[2]);
  inline double phi(float xy1[2], float xy2[2]);

  //  calculate theta from x, y, z
  inline double theta(double x, double y, double z);
  //  calculate theta for a line defined by xyz1 and xyz2 (xyz2-xyz1)
  inline double theta(double xyz1[3], double xyz2[3]);
  inline double theta(float xyz1[3], float xyz2[3]);
  //  calculate theta from eta
  inline double theta(double etap);

  //  calculate eta from x, y, z (return also theta)
  inline double eta(double x, double y, double z);
  //  calculate eta for a line defined by xyz1 and xyz2 (xyz2-xyz1)
  inline double eta(double xyz1[3], double xyz2[3]);
  inline double eta(float xyz1[3], float xyz2[3]);
  //  calculate eta from theta
  inline double eta(double th);

  // calculate rapidity from E, pz
  inline double y(double E, double pz);

  //  calculate phi1-phi2 keeping value between 0 and pi
  inline double delta_phi(double ph11, double phi2);
  // calculate phi1-phi2 keeping value between -pi and pi
  inline double signed_delta_phi(double ph11, double phi2);
  // calculate eta1 - eta2
  inline double delta_eta(double eta1, double eta2);

  //  calculate deltaR
  inline double delta_R(double eta1, double phi1, double eta2, double phi2);

  //  calculate unit vectors given two points
  inline void uvectors(double u[3], double xyz1[3], double xyz2[3]);
  inline void uvectors(float u[3], float xyz1[3], float xyz2[3]);

  inline double tanl_from_theta(double theta);
  inline double theta_from_tanl(double tanl); 
}

inline
double kinem::phi(double x, double y)
{
  double PHI=atan2(y, x);
  return (PHI>=0)? PHI : kinem::TWOPI+PHI;
}

inline 
double kinem::phi(float xy1[2], float xy2[2]){
    double dxy1[2]={xy1[0],xy1[1]};
    double dxy2[2]={xy2[0],xy2[1]};
    return phi(dxy1,dxy2);
  }

inline
double kinem::phi(double xy1[2], double xy2[2])
{
  double x=xy2[0]-xy1[0];
  double y=xy2[1]-xy1[1];
  return phi(x, y);
}

inline
double kinem::delta_phi(double phi1, double phi2)
{
  double PHI=fabs(phi1-phi2);
  return (PHI<=PI)? PHI : kinem::TWOPI-PHI;
}

inline
double kinem::delta_eta(double eta1, double eta2)
{
  return eta1 - eta2;
}

inline
double kinem::signed_delta_phi(double phi1, double phi2)
{
  double phia=phi1;
  if(phi1>PI) phia=phi1-kinem::TWOPI;
  double phib=phi2;
  if(phi2>PI) phib=phi2-kinem::TWOPI;
  double dphi=phia-phib;
  if(dphi>PI) dphi-=kinem::TWOPI;
  if(dphi<-PI) dphi+=kinem::TWOPI;
  return dphi;
}

inline double kinem::delta_R(double eta1, double phi1, double eta2, double phi2)
{
   double deta = eta1-eta2;
   double dphi = kinem::delta_phi(phi1,phi2);
   return sqrt(deta*deta + dphi*dphi);
}

inline
double kinem::theta(double xyz1[3], double xyz2[3])
{
  double x=xyz2[0]-xyz1[0];
  double y=xyz2[1]-xyz1[1];
  double z=xyz2[2]-xyz1[2];
  return theta(x, y, z);
}

inline 
double kinem::theta(float xyz1[3], float xyz2[3]){
    double dxyz1[3]={xyz1[0],xyz1[1],xyz1[2]};
    double dxyz2[3]={xyz2[0],xyz2[1],xyz2[2]};
    return theta(dxyz1,dxyz2);
}

inline 
double kinem::theta(double x, double y, double z)
{
  return atan2(sqrt(x*x + y*y), z);
}

inline 
double kinem::theta(double etap)
{
  return 2.0*atan(exp(-etap));
}

inline
double kinem::eta(double xyz1[3], double xyz2[3])
{
  double x=xyz2[0]-xyz1[0];
  double y=xyz2[1]-xyz1[1];
  double z=xyz2[2]-xyz1[2];
  return eta(x, y, z);
}

inline
double kinem::eta(float xyz1[3], float xyz2[3]){
    double dxyz1[3]={xyz1[0],xyz1[1],xyz1[2]};
    double dxyz2[3]={xyz2[0],xyz2[1],xyz2[2]};
    return eta(dxyz1,dxyz2);
}

inline
double kinem::eta(double x, double y, double z)
{
  return 0.5*log((sqrt(x*x + y*y + z*z) + z + EPSILON) / 
                 (sqrt(x*x + y*y + z*z) - z + EPSILON));
}

inline
double kinem::eta(double th)
{
  if(th == 0) return ETA_LIMIT;
  if(th >= PI-0.0001) return -ETA_LIMIT;
  return -log(tan(th/2.0));
}

inline 
double kinem::y(double E, double pz)
{
  return 0.5 * log ((E+pz+EPSILON)/(E-pz+EPSILON));
}

inline
void kinem::uvectors(double u[3], double xyz1[3], double xyz2[3]){
  double xdiff=xyz2[0]-xyz1[0];
  double ydiff=xyz2[1]-xyz1[1];
  double zdiff=xyz2[2]-xyz1[2];
  double s=sqrt(xdiff*xdiff+ydiff*ydiff+zdiff*zdiff);
  if(s > 0) {
    u[0]=xdiff/s;
    u[1]=ydiff/s;
    u[2]=zdiff/s;
  }else{
    u[0]=0;
    u[1]=0;
    u[2]=0;
  }
}

inline 
void kinem::uvectors(float u[3], float xyz1[3], float xyz2[3]){
    double du[3];
    double dxyz1[3]={xyz1[0],xyz1[1],xyz1[2]};
    double dxyz2[3]={xyz2[0],xyz2[1],xyz2[2]};
    uvectors(du,dxyz1,dxyz2);
    u[0]=du[0];
    u[1]=du[1];
    u[2]=du[2];
}

inline
double kinem::tanl_from_theta(double theta){
  return tan(PI/2.0 - theta);
}

inline
double kinem::theta_from_tanl(double tanl){
  return PI/2.0 - atan(tanl);
}

#endif // INC_ANGLESUTIL
