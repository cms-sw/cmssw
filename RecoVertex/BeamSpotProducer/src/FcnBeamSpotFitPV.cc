#include "RecoVertex/BeamSpotProducer/interface/FcnBeamSpotFitPV.h"
#include "Math/SVector.h"
#include "Math/SMatrix.h"
// #include "Math/SMatrixDfwd.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

//
// constructor from vertex data
//
FcnBeamSpotFitPV::FcnBeamSpotFitPV(const vector<BeamSpotFitPVData>& data) : 
  data_(data), errorDef_(1.) { 
  //
  // default: no additional cuts
  //
  lowerLimits_[0] = lowerLimits_[1] = lowerLimits_[2] = -1.e30;
  upperLimits_[0] = upperLimits_[1] = upperLimits_[2] =  1.e30;
}

void 
FcnBeamSpotFitPV::setLimits (float xmin, float xmax,
			  float ymin, float ymax,
			  float zmin, float zmax) 
{
  lowerLimits_[0] = xmin;
  lowerLimits_[1] = ymin;
  lowerLimits_[2] = zmin;
  upperLimits_[0] = xmax;
  upperLimits_[1] = ymax;
  upperLimits_[2] = zmax;
}

unsigned int
FcnBeamSpotFitPV::nrOfVerticesUsed () const
{
  //
  // count vertices imposing the current limits
  //
  unsigned int nVtx = 0;
  double v1(0);
  double v2(0);
  double v3(0);
  for ( vector<BeamSpotFitPVData>::const_iterator ipv=data_.begin();
	ipv!=data_.end(); ++ipv ) {
    v1 = (*ipv).position[0];
    if ( v1<lowerLimits_[0] || v1>upperLimits_[0] )  continue;
    v2 = (*ipv).position[1];
    if ( v2<lowerLimits_[1] || v2>upperLimits_[1] )  continue;
    v3 = (*ipv).position[2];
    if ( v3<lowerLimits_[2] || v3>upperLimits_[2] )  continue;

    ++nVtx;
  }

  return nVtx;
}

double
FcnBeamSpotFitPV::operator() (const std::vector<double>& pars) const
{
  //
  // fit parameters
  //
  double vb1 = pars[0];
  double vb2 = pars[1];
  double vb3 = pars[2];
  double sigb1 = pars[3]; 
  double corrb12 = pars[4];
  double sigb2 = pars[5];
  double dxdz = pars[6];
  double dydz = pars[7];
  double sigb3 = pars[8];
  double escale = pars[9];
  //
  // covariance matrix of the beamspot distribution
  //
  typedef ROOT::Math::SVector<double,3> Vector3D;
  typedef ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> > Matrix3D;
  Matrix3D covb;
  double varb1 = sigb1*sigb1;
  double varb2 = sigb2*sigb2;
  double varb3 = sigb3*sigb3;
// parametrisation: rotation (dx/dz, dy/dz); covxy
  covb(0,0) = varb1;
  covb(1,0) = covb(0,1) = corrb12*sigb1*sigb2;
  covb(1,1) = varb2;
  covb(2,0) = covb(0,2) = dxdz*(varb3-varb1)-dydz*covb(1,0);
  covb(2,1) = covb(1,2) = dydz*(varb3-varb2)-dxdz*covb(1,0);
  covb(2,2) = varb3;

  //
  // calculation of the likelihood function
  //
  double sumLL(0);
  double v1(0);
  double v2(0);
  double v3(0);
  double ev1(0);
  double corr12(0);
  double ev2(0);
  double corr13(0);
  double corr23(0);
  double ev3(0);
  //
  // vertex - beamspot difference and total covariance matrix
  //
  Vector3D dv;
  Matrix3D cov;
  Matrix3D wgt;
  //
  // iteration over vertices
  //
  for ( vector<BeamSpotFitPVData>::const_iterator ipv=data_.begin();
	ipv!=data_.end(); ++ipv ) {
    //
    // additional selection
    //
    v1 = (*ipv).position[0];
    if ( v1<lowerLimits_[0] || v1>upperLimits_[0] )  continue;
    v2 = (*ipv).position[1];
    if ( v2<lowerLimits_[1] || v2>upperLimits_[1] )  continue;
    v3 = (*ipv).position[2];
    if ( v3<lowerLimits_[2] || v3>upperLimits_[2] )  continue;
    //
    // vertex errors (after scaling) and correlations
    //
    ev1 = (*ipv).posError[0];
    corr12 = (*ipv).posCorr[0];
    ev2 = (*ipv).posError[1];
    corr13 = (*ipv).posCorr[1];
    corr23 = (*ipv).posCorr[2];
    ev3 = (*ipv).posError[2];
    ev1 *= escale;
    ev2 *= escale;
    ev3 *= escale;
    //
    // vertex covariance matrix
    //
    cov(0,0) = ev1*ev1;
    cov(1,0) = cov(0,1) = ev1*ev2*corr12;
    cov(1,1) = ev2*ev2;
    cov(2,0) = cov(0,2) = ev1*ev3*corr13;
    cov(2,1) = cov(1,2) = ev2*ev3*corr23;
    cov(2,2) = ev3*ev3;
    //
    // total covariance and weight matrix
    //
    cov += covb;
    int ifail;
    wgt = cov.Inverse(ifail);
    if ( ifail ) {
      //edm::LogWarning("FcnBeamSpotFitPV") 
      cout << "Inversion failed" << endl;
      return -1.e30;
    }
    //
    // difference of vertex and beamspot positions
    //
    dv(0) = v1 - vb1;
    dv(1) = v2 - vb2;
    dv(2) = v3 - vb3;
    //
    // exponential term and normalization
    // (neglecting the additional cut)
    //
    sumLL += ROOT::Math::Similarity(dv,wgt);
    double det;
    cov.Det2(det);
    sumLL += log(det);
  }

  return sumLL;
}

