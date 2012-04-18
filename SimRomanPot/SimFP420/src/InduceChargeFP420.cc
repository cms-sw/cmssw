///////////////////////////////////////////////////////////////////////////////
// File: InduceChargeFP420
// Date: 08.2008
// Description: InduceChargeFP420 for FP420
// Modifications:  
///////////////////////////////////////////////////////////////////////////////
//#include "SimRomanPot/SimFP420/interface/SimRPUtil.h"
#include "SimRomanPot/SimFP420/interface/InduceChargeFP420.h"
#include <gsl/gsl_sf_erf.h>
#include<iostream>
using namespace std;

static float capacitive_coupling[2] =  {.80,.10};
static float FiducialXYZ[3] =  {3.6,4.,0.250};// dX/2, dY/2,dZ -- mm fiducial dimension of Si plate

IChargeFP420::hit_map_type InduceChargeFP420::induce(CDrifterFP420::collection_type _collection_points, int numStrips, double localPitch, int numStripsW, double localPitchW, int xytype, int verbosity){
  signalCoupling.clear();
  signalCoupling.push_back(capacitive_coupling[0]);
  signalCoupling.push_back(capacitive_coupling[1]);
  
  // in mm local coordinates   (temporarily)
  // float FiducialX = 5., FiducialY = 10., FiducialZ = 0.250;
  int stripLeft, stripRight, stripLeftW, stripRightW;
  double upperBound, lowerBound, upperBoundW, lowerBoundW;
  
  
  hit_map_type hit_signal;
  
  // map to store pixel integrals in the x and in the y directions
  std::map<int, float, less<int> > x,y; 
  
  
  for (CDrifterFP420::collection_type::const_iterator sp=_collection_points.begin();  sp != _collection_points.end(); sp++ ){
    
    float chargePositionW=-1.; // charge in strip coord in Wide pixel
    float chargePosition=-1.; // charge in strip coord
    
    // define chargePosition
    G4ThreeVector Position3D = (*sp).position(); // charge in strip coord
    
    if(verbosity>0) {
      std::cout << " =============================*InduceChargeFP420:induce:Position3D= " << Position3D << std::endl;
      std::cout << " localPitch= " << localPitch << std::endl;
      std::cout << " xytype= " << xytype << " numStrips= " << numStrips << std::endl;
    }
    
    
    
    // is slice still inside fiducial volume of the plate? if not ->  put slice energy to zero.
    if( abs(Position3D.x())<FiducialXYZ[0] && 
        abs(Position3D.y())<FiducialXYZ[1] ) {
      if( abs(Position3D.z())<FiducialXYZ[2] ) {
      }
      else{
        //	(*sp).amplitude() == 0.;
	std::cout << " *InduceChargeFP420:Z slice outside the plate: Position3D= " << Position3D << std::endl;
      }
    }
    else{
      //      (*sp).amplitude() == 0.;
      std::cout << " *InduceChargeFP420:XY slice outside the plate: Position3D= " << Position3D << std::endl;
    }
    
    // chargePosition - still local coordinates, so exchange x and y due to 90 degree rotation
    // Yglobal::
    if(xytype == 1) {
      // = 
      chargePosition = 0.5*numStrips + Position3D.x()/localPitch ;// charge in strip coord. in l.r.f starting at edge of plate
      chargePositionW = 0.5*numStripsW + Position3D.y()/localPitchW ;// charge in strip coord. in l.r.f starting at edge of plate
    }
    //X:
    else if(xytype == 2) {
      // = 
      chargePosition = 0.5*numStrips + Position3D.y()/localPitch ;// charge in strip coord. in l.r.f starting at edge of plate
      chargePositionW = 0.5*numStripsW + Position3D.x()/localPitchW ;// charge in strip coord. in l.r.f starting at edge of plate
    }
    else {
      std::cout << "**** InduceChargeFP420:  !!!  ERROR: you have not to be here !!!  xytype=" << xytype << std::endl;
      //     break;
    }
    
    if(verbosity>0) {
      if(xytype==2){
	std::cout << "===================================**** InduceChargeFP420: xytype= " << xytype << std::endl;
	std::cout << "  chargePositionW= " << chargePositionW << "  chargePosition= " << chargePosition << std::endl;
	std::cout << "Position3D= " << Position3D << std::endl;
      }
    }
    
    float chargeSpread = (*sp).sigma()/localPitch ;  // sigma in strip pitches
    float chargeSpreadW = (*sp).sigma()/localPitchW ;  // sigma in strip pitches
    
    // Define strips intervals along x: check edge condition
    
    stripLeft  = int( chargePosition-clusterWidth*chargeSpread);
    stripRight = int( chargePosition+clusterWidth*chargeSpread);
    stripLeft  = (0<stripLeft ? stripLeft : 0);
    stripRight = (numStrips >stripRight ? stripRight : numStrips-1);
    
    stripLeftW  = int( chargePositionW-clusterWidth*chargeSpreadW);
    stripRightW = int( chargePositionW+clusterWidth*chargeSpreadW);
    stripLeftW  = (0<stripLeftW ? stripLeftW : 0);
    stripRightW = (numStripsW >stripRightW ? stripRightW : numStripsW-1);
    
    if(verbosity>1) {
      std::cout << " Position3D =  " << Position3D << "amplitude=" << (*sp).amplitude() << std::endl;
      std::cout << " MaxChargeSpread= " << clusterWidth*chargeSpread << " sigma= " << (*sp).sigma() << std::endl;
      std::cout << "*** numStrips= " << numStrips << " localPitch= " << localPitch << "xytype=" << xytype << std::endl;
      std::cout << " chargePosition= " << chargePosition << " chargeSpread= " << chargeSpread << std::endl;
      std::cout << " stripLeft= " << stripLeft << " stripRight= " << stripRight << std::endl;
      std::cout << " chargePositionW= " << chargePositionW << " chargeSpreadW= " << chargeSpreadW << std::endl;
      std::cout << " stripLeftW= " << stripLeftW << " stripRightW= " << stripRightW << std::endl;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// X
    for (int i=stripLeft; i<=stripRight; i++){
      /* Definition of the integration borns */
      // go to "left"
      if (i == 0) lowerBound = 0. ;
      else {
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePosition)/chargeSpread, &result);
	if (status != 0) std::cerr<<"InduceChargeFP420::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	lowerBound = 1. - result.val;
	if(verbosity>0) {
	  std::cout << "go to left: i=  " << i << "lowerBound=" << lowerBound << std::endl;
	}
      }
      
      // go to "right"
      if (i == numStrips-1) upperBound = 1.;
      else {
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePosition+1)/chargeSpread, &result);
	if (status != 0) std::cerr<<"InduceChargeFP420::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	upperBound = 1. - result.val;
	
	if(verbosity>0) {
	  std::cout << "go to right: i=  " << i << "upperBound=" << upperBound << std::endl;
	}
      }
      
      double totalIntegrationRange = upperBound - lowerBound;
      x[i] = totalIntegrationRange; // save strip integral 
      
      if(totalIntegrationRange<=0.) std::cout << " upperBound= " << upperBound << " lowerBound= " << lowerBound << std::endl;
      if(verbosity==-30) {
	std::cout << " *InduceChargeFP420:====================================X i =  " << i << std::endl;
	std::cout << " upperBound= " << upperBound << " lowerBound= " << lowerBound << std::endl;
	std::cout << " totalIntegrationRange= " << totalIntegrationRange << std::endl;
	std::cout << " *InduceChargeFP420:==================================== " << std::endl;
      }
      
    }// for
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// XW=Y
    for (int i=stripLeftW; i<=stripRightW; i++){
      /* Definition of the integration borns */
      // go to "left"
      if (i == 0) lowerBoundW = 0. ;
      else {
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePositionW)/chargeSpreadW, &result);
	if (status != 0) std::cerr<<"InduceChargeFP420::W could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	lowerBoundW = 1. - result.val;
	
	
	if(verbosity>0) {
	  std::cout << "go to left: i=  " << i << "lowerBoundW=" << lowerBoundW << std::endl;
	}
      }
      
      // go to "right"
      if (i == numStripsW-1) upperBoundW = 1.;
      else {
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePositionW+1)/chargeSpreadW, &result);
	if (status != 0) std::cerr<<"InduceChargeFP420::W could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	upperBoundW = 1. - result.val;
	
	if(verbosity>0) {
	  std::cout << "go to right: i=  " << i << "upperBoundW=" << upperBoundW << std::endl;
	}
      }
      
      double totalIntegrationRange = upperBoundW - lowerBoundW;
      y[i] = totalIntegrationRange; // save W strip integral 
      
      if(totalIntegrationRange<=0.) std::cout << " upperBoundW= " << upperBoundW << " lowerBoundW= " << lowerBoundW << std::endl;
      
      if(verbosity==-30) {
	std::cout << " *InduceChargeFP420:====================================XW  i= " << i << std::endl;
	std::cout << " upperBoundW= " << upperBoundW << " lowerBoundW= " << lowerBoundW << std::endl;
	std::cout << " totalIntegrationRange= " << totalIntegrationRange << std::endl;
	std::cout << " *InduceChargeFP420:==================================== " << std::endl;
      }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //calculate signal on x strips with including capacitive coupling
    int nSignalCoupling = signalCoupling.size();
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    int lll,copyinlll;
    lll = unpackLayerIndex(rn0,zside);
    if(lll==1) {copyinlll=  lll/2;}
    else if(lll==2) {copyinlll=  (lll-1)/2;}
    else{std::cout << " InduceChargeFP420:WARNING plane number in superlayer= " << lll << std::endl;}
*/    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if(verbosity>1) {
      std::cout << "InduceChargeFP420:   *************************************************************** " << std::endl;
      std::cout << " numStripsW= " << numStripsW << " numStrips= " << numStrips << std::endl;
      std::cout << " nSignalCoupling= " << nSignalCoupling << " xytype= " << xytype << std::endl;
      std::cout << " stripLeftW= " << stripLeftW << " stripRightW= " << stripRightW << std::endl;
      std::cout << " stripLeft= " << stripLeft << " stripRight= " << stripRight << std::endl;
    }
    // Get the 2D charge integrals by folding x and y strips
    for (int iy=stripLeftW; iy<=stripRightW; iy++){ // loop over Wide y index
      for (int ix=stripLeft; ix<=stripRight; ix++){ // loop over x index
	for (int k = -nSignalCoupling+1 ; k<=nSignalCoupling-1 ; k++) {
	  if (ix+k >= 0 && ix+k < numStrips ) { 
	    float ChargeFraction = signalCoupling[abs(k)]*(x[ix]*y[iy]/geVperElectron)*(*sp).amplitude();
	    if( ChargeFraction > 0. ) {
	      //  int chan = PixelDigi::pixelToChannel( ix, iy);  // Get index 
	      int chan = iy*numStrips + (ix+k) ;  // Get index 
	      
	      //    if(k==0 ){
	      // 	std::cout << "InduceChargeFP420:                                              chan= " << chan << std::endl;
	      // 	std::cout << "ix= " << ix << "iy= " << iy << std::endl;
	      //  }
	      if(verbosity>0) {
		if(k==0 && xytype==2){
		  std::cout << "InduceChargeFP420:                                              chan= " << chan << std::endl;
		  std::cout << "ix= " << ix << "iy= " << iy << "k= " << k << "ChargeFraction= " << ChargeFraction << std::endl;
		  std::cout << "hit_signal[chan]= " << hit_signal[chan] << "geVperElectron= " << geVperElectron << std::endl;
		  std::cout << "signalCoupling[abs(k)]= " << signalCoupling[abs(k)] << "x[ix]= " << x[ix] << "y[iy]= " << y[iy] << "(*sp).amplitude()= " << (*sp).amplitude() << std::endl;
		}
	      }
	      // Load the amplitude:
	      hit_signal[chan] += ChargeFraction;
	    } // endif ChargeFraction
	  } // endif ix+k
	  else{
	    //std::cout << "WARNING:                         ix+k =" << ix+k << std::endl;
	  }// endif ix+k
	} // endfor k
      } //endfor ix
    } //endfor iy
    
    if(verbosity>0) {
      std::cout << "================================================================================= " << std::endl;
    }
    
    
    
  } //for loop on ions (*sp)
  
  if(verbosity>0) {
    std::cout << "end of InduceChargeFP420============================= " << std::endl;
  }
  
  
  return hit_signal;
  
  
}
