///////////////////////////////////////////////////////////////////////////////
// File: InduceChargeFP420
// Date: 12.2006
// Description: InduceChargeFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimRomanPot/SimFP420/interface/InduceChargeFP420.h"
#include <gsl/gsl_sf_erf.h>
#include<iostream>
using namespace std;
//#define mydigidebug4

static float capacitive_coupling[2] =  {.80,.10};
static float FiducialXYZ[3] =  {5.,10.,0.250};// dX/2, dY/2,dZ -- mm fiducial dimension of Si plate

IChargeFP420::hit_map_type InduceChargeFP420::induce(CDrifterFP420::collection_type _collection_points, int numStrips, double localPitch, float chargePositionOld, int zside){
  signalCoupling.clear();
  signalCoupling.push_back(capacitive_coupling[0]);
  signalCoupling.push_back(capacitive_coupling[1]);

  // in mm local coordinates   (temporarily)
  // float FiducialX = 5., FiducialY = 10., FiducialZ = 0.250;
  int stripLeft, stripRight;
  double upperBound, lowerBound;

    
  hit_map_type hit_signal;
    
  for (CDrifterFP420::collection_type::const_iterator sp=_collection_points.begin();  sp != _collection_points.end(); sp++ ){

    float chargePosition; // charge in strip coord
    chargePosition = chargePositionOld;
    // define chargePosition
    G4ThreeVector Position3D = (*sp).position(); // charge in strip coord

#ifdef mydigidebug4
   std::cout << " =============================*InduceChargeFP420:induce:Position3D= " << Position3D << std::endl;
   std::cout << " chargePositionOld= " << chargePositionOld << " localPitch= " << localPitch << std::endl;
   std::cout << " zside= " << zside << " numStrips= " << numStrips << std::endl;
#endif



    // is slice still inside fiducial volume of the plate? if not ->  put slice energy to zero.
    if( abs(Position3D.x())<FiducialXYZ[0] && 
        abs(Position3D.y())<FiducialXYZ[1] ) {
      if( abs(Position3D.z())<FiducialXYZ[2] ) {
      }
      else{
	(*sp).amplitude() == 0.;
//#ifdef mydigidebug4
std::cout << " *InduceChargeFP420:Z slice outside the plate: Position3D= " << Position3D << std::endl;
//#endif
      }
    }
    else{
      (*sp).amplitude() == 0.;
//#ifdef mydigidebug4
	std::cout << " *InduceChargeFP420:XY slice outside the plate: Position3D= " << Position3D << std::endl;
//#endif
    }

    // chargePosition - still local coordinates, so exchange x and y due to 90 degree rotation
    // Yglobal::
    if(zside == 1) {
      // = 
      chargePosition = 0.5*numStrips + Position3D.x()/localPitch ;// charge in strip coord. in l.r.f starting at edge of plate
    }
    //X:
    else if(zside == 2) {
      // = 
      chargePosition = 0.5*numStrips + Position3D.y()/localPitch ;// charge in strip coord. in l.r.f starting at edge of plate
    }
    else {
      std::cout << "**** InduceChargeFP420:  !!!  ERROR: you have not to be here !!!  zside=" << zside << std::endl;
      //     break;
    }

#ifdef mydigidebug4
    //    if(chargePositionOld != chargePosition && zside == 1 && chargePosition<200 ) {
   std::cout << " =========================================================================== " << std::endl;
      std::cout << "**** InduceChargeFP420: zside= " << zside << std::endl;
      std::cout << "chargePositionOld= " << chargePositionOld << "  chargePosition= " << chargePosition << std::endl;
      std::cout << "Position3D= " << Position3D << std::endl;
      //    }
#endif

    float chargeSpread = (*sp).sigma()/localPitch ;  // sigma in strip pitches
    
    // Define strips intervals along x: check edge condition
    
    stripLeft  = int( chargePosition-clusterWidth*chargeSpread);
    stripRight = int( chargePosition+clusterWidth*chargeSpread);
    stripLeft  = (0<stripLeft ? stripLeft : 0);
    stripRight = (numStrips >stripRight ? stripRight : numStrips-1);
     
#ifdef mydigidebug4
   std::cout << " Position3D =  " << Position3D << "amplitude=" << (*sp).amplitude() << std::endl;
   std::cout << " chargePosition= " << chargePosition << std::endl;
   std::cout << " MaxChargeSpread= " << clusterWidth*chargeSpread << std::endl;
   std::cout << " chargeSpread= " << chargeSpread << " sigma= " << (*sp).sigma() << std::endl;
   std::cout << " stripLeft= " << stripLeft << " stripRight= " << stripRight << std::endl;
   std::cout << " numStrips= " << numStrips << " localPitch= " << localPitch << "zside=" << zside << std::endl;
#endif
    for (int i=stripLeft; i<=stripRight; i++){

      /* Definition of the integration borns */

 // go to "left"
      if (i == 0) lowerBound = 0. ;
      else {
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePosition)/chargeSpread, &result);
	if (status != 0) cerr<<"GaussNoiseProducerFP420::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	lowerBound = 1. - result.val;

#ifdef mydigidebug4
   std::cout << "go to left: i=  " << i << "lowerBound=" << lowerBound << std::endl;
#endif
      }

 // go to "right"
      if (i == numStrips-1) upperBound = 1.;
      else {
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePosition+1)/chargeSpread, &result);
        if (status != 0) cerr<<"GaussNoiseProducerFP420::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	upperBound = 1. - result.val;

#ifdef mydigidebug4
   std::cout << "go to right: i=  " << i << "upperBound=" << upperBound << std::endl;
#endif
      }
       


      double totalIntegrationRange = upperBound - lowerBound;
      if(totalIntegrationRange<=0.) std::cout << " upperBound= " << upperBound << " lowerBound= " << lowerBound << std::endl;
     
      //calculate signal on strips including capacitive coupling
     
      int nSignalCoupling = signalCoupling.size();

#ifdef mydigidebug4
   std::cout << " *InduceChargeFP420:induce:loops on collection_points and then strip i= " << i << std::endl;
   std::cout << " upperBound= " << upperBound << " lowerBound= " << lowerBound << std::endl;
   std::cout << " totalIntegrationRange= " << totalIntegrationRange << " nSignalCoupling= " << nSignalCoupling << std::endl;
   std::cout << " *InduceChargeFP420:==================================== " << std::endl;
#endif
   for (int j = -nSignalCoupling+1 ; j<=nSignalCoupling-1 ; j++) {
     if (i+j >= 0 && i+j < numStrips ) { 
       hit_signal[i+j] += signalCoupling[abs(j)]* 
	 (totalIntegrationRange/geVperElectron)*
	 (*sp).amplitude();   
       
#ifdef mydigidebug4
       //    if(zside == 1 ) {
       std::cout << " *InduceChargeFP420:=========== zside=" << zside << std::endl;
       std::cout << "chargePosOld= " << chargePositionOld << " chargePos= " << chargePosition << std::endl;
       std::cout << "Position3D= " << Position3D << std::endl;
       std::cout << " i+j= " << i+j << " hit_signal[i+j]= " << hit_signal[i+j] << std::endl;
       std::cout << " (*sp).amplitude()= " << (*sp).amplitude() << " i= " << i << " j= " << j << std::endl;
       std::cout << " upperBound= " << upperBound << " lowerBound= " << lowerBound << std::endl;
       std::cout << " signalCoupling[abs(j)]= " << signalCoupling[abs(j)] << std::endl;
       std::cout << " ===================== " << std::endl;
       //    } 
#endif
       
       
     } // if
     else{
#ifdef mydigidebug4
       std::cout << "Inducecheck: i+j= " << i+j << " numStrips= " << numStrips  <<" i= " << i << std::endl;
       std::cout << " ====== " << std::endl;
#endif
     } // if
   } //for loop on j
#ifdef mydigidebug4
       std::cout << "Inducecheck:                           end loop on j charge spread " << std::endl;
#endif
   
    } //for loop on i strips in intergation
#ifdef mydigidebug4
       std::cout << "Inducecheck:                           end loop on i strips in intagration " << std::endl;
#endif
    
  } //for loop on ions (*sp)



  return hit_signal;


}
