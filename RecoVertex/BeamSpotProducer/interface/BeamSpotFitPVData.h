#ifndef BeamSpotFitPVData_
#define BeamSpotFitPVData_
/** \class BeamSpotFitPVData
 *  Simple structure to hold vertex position and covariance.
 *  \author WA, 9/3/2010
 */
struct BeamSpotFitPVData {
  float bunchCrossing; // bunch crossing
  float position[3]; //< x, y, z position
  float posError[3]; //< x, y, z error
  float posCorr[3];  //< xy, xz, yz correlations
};
#endif
