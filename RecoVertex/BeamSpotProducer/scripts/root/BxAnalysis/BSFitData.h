#ifndef BSFitData_
#define BSFitData_
/** \class BSFitPVData
 *  Simple structure to hold vertex position and covariance.
 *  \author WA, 9/3/2010
 */
struct BSFitData {
  int bunchCrossing;    // bunch crossing
  float xyz[3];           //< x, y, z position
  float xyzErr[3];        //< x, y, z error
  float xyzwidth[3];      //< xy, xz, yz correlations
  float xyzwidthErr[3]; 
};
#endif
