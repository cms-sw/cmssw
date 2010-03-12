#ifndef FcnBeamSpotFitPV_h_
#define FcnBeamSpotFitPV_h_
/** \class FcnBeamSpotFitPV
 *  LogLH function for 3D beam spot fit from primary vertex data. Assumes
 *  a Gaussian resolution function for the vertices provided in the
 *  form of a vector of BeamSpotFitPVData and a Gaussian beam profile
 *  in 3D: based on a covariance without xz and yz correlations +
 *  linearized rotations in the xz and yz planes. 
 *  Parameters: x, y, z, ex, corrxy, ey, dxdz, dydz, ez, scale
 *   x, y, z ...... beamspot position
 *   ex, ey, ez ... beamspot width
 *   corrxy ....... xy correlation in the system of the beam line
 *   dxdz, dydz ... rotations in xz and yz
 *   scale ........ scaling factor for vertex errors
 *   \author WA, 9/3/2010
 */
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotFitPVData.h"
#include "Minuit2/FCNBase.h"

#include <vector> 

class FcnBeamSpotFitPV : public ROOT::Minuit2::FCNBase { 
public: 
  // constructor from vertex data
  FcnBeamSpotFitPV(const std::vector<BeamSpotFitPVData>& data);
  ~FcnBeamSpotFitPV() {} 
  // additional vertex selection using limits in x, y, z
  void setLimits (float xmin, float xmax,
		  float ymin, float ymax,
		  float zmin, float zmax);
  // deltaFcn for definition of the uncertainty
  double Up() const {return errorDef_;} 
  // -2lnL value based on vector of parameters
  double operator() (const std::vector<double>&) const; 
  // vertex count used for the fit (after selection)
  unsigned int nrOfVerticesUsed () const;
private: 
  const std::vector<BeamSpotFitPVData>& data_; //< vertex data
  double errorDef_;                            //< error definition for Minuit

  float lowerLimits_[3];                       //< lower limits for x,y,z
  float upperLimits_[3];                       //< upper limits for x,y,z
}; 
#endif
