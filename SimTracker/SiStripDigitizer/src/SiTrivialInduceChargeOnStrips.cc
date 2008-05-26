#include "SimTracker/SiStripDigitizer/interface/SiTrivialInduceChargeOnStrips.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include <Math/ProbFuncMathCore.h>
#include<iostream>

SiTrivialInduceChargeOnStrips::SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g):conf_(conf)
{
  peak = conf_.getParameter<bool>("APVpeakmode");
  // Coupling Constant
  //  signalCoupling.clear();
  signalCoupling_TIB.clear();
  signalCoupling_TID.clear();
  signalCoupling_TOB.clear();
  signalCoupling_TEC.clear();
  if(peak){
    // TIB
    signalCoupling_TIB = conf_.getParameter<std::vector<double> >("CouplingCostantPeakTIB");
    // TID
    signalCoupling_TID = conf_.getParameter<std::vector<double> >("CouplingCostantPeakTID");
    // TOB
    signalCoupling_TOB = conf_.getParameter<std::vector<double> >("CouplingCostantPeakTOB");
    // TEC
    signalCoupling_TEC = conf_.getParameter<std::vector<double> >("CouplingCostantPeakTEC");
    //
  }else{
    // TIB
    signalCoupling_TIB = conf_.getParameter<std::vector<double> >("CouplingCostantDecTIB");
    // TID
    signalCoupling_TID = conf_.getParameter<std::vector<double> >("CouplingCostantDecTID");
    // TOB
    signalCoupling_TOB = conf_.getParameter<std::vector<double> >("CouplingCostantDecTOB");
    // TEC
    signalCoupling_TEC = conf_.getParameter<std::vector<double> >("CouplingCostantDecTEC");
    //
  }
  clusterWidth = 3.; 
  geVperElectron = g;
}
void SiTrivialInduceChargeOnStrips::induce(SiChargeCollectionDrifter::collection_type _collection_points, const StripGeomDetUnit& det, 
					   std::vector<double>& locAmpl, unsigned int& minCha, unsigned int& maxCha){
  //  const StripTopology& t = dynamic_cast<const StripTopology&>(det.topology());
  const StripTopology& t = dynamic_cast<const StripTopology&>(det.specificTopology()); // AG
  
  int numStrips = t.nstrips();
  int stripLeft, stripRight;
  double upperBound, lowerBound;
  
  // fill local Amplitudes with zeroes
  // it is done also in SiStripDigitizerAlgorithm 'ma non si sa mai'
  //  std::fill(locAmpl.begin(),locAmpl.end(),0.);
  //
  
  minCha=locAmpl.size();
  maxCha=0;
  for (SiChargeCollectionDrifter::collection_type::const_iterator sp=_collection_points.begin();  sp != _collection_points.end(); sp++ ){
    float chargePosition = t.strip((*sp).position()); // charge in strip coord
    float localPitch = t.localPitch((*sp).position()); // local strip pitch 
    float chargeSpread = (*sp).sigma()/localPitch ;  // sigma in strip coord
    
    // Define strips intervals along x: check edge condition
    
    stripLeft  = int( chargePosition-clusterWidth*chargeSpread);
    stripRight = int( chargePosition+clusterWidth*chargeSpread);
    stripLeft  = (0<stripLeft ? stripLeft : 0);
    stripRight = (numStrips >stripRight ? stripRight : numStrips-1);
   

    for (int i=stripLeft; i<=stripRight; i++){
      /* Definition of the integration borns */
      if (i == 0) lowerBound = 0. ;
      else {
        lowerBound = 1. - ROOT::Math::normal_cdf_c(i,chargeSpread,chargePosition);
      }
      if (i == numStrips-1) upperBound = 1.;
      else {
        upperBound = 1. - ROOT::Math::normal_cdf_c(i+1,chargeSpread,chargePosition);
      }
       
      double totalIntegrationRange = upperBound - lowerBound;
      
      //calculate signal on strips including capacitive coupling
      
      // Choice of the subdtector charge coupling
      std::vector<double>* signalCoupling = 0;
      int subDet_enum=det.specificType().subDetector();
      switch (subDet_enum) {
      case GeomDetEnumerators::TIB:
	{
	  signalCoupling = &(signalCoupling_TIB);
	  break;
	}
      case GeomDetEnumerators::TOB:
	{
	  signalCoupling = &(signalCoupling_TOB);
	  break;
	}
      case GeomDetEnumerators::TID:
	{
	  signalCoupling = &(signalCoupling_TID);
	  break;
	}
      case GeomDetEnumerators::TEC:
	{
	  signalCoupling = &(signalCoupling_TEC);
	  break;
	}
      default: 
	{
	  std::cout << "SiTrivialInduceChargeOnStrips ERROR - Not a Tracker Subdetector " << subDet_enum << std::endl;
	  break;
	}
      } // switch
      
      int nSignalCoupling = (*signalCoupling).size();
      
      // debug
      std::cout << "SiTrivialInduceChargeOnStrips ChargeCoupling for subdetector " << subDet_enum << std::endl;
      for(unsigned int iCC=0; iCC!=nSignalCoupling ; iCC++)
	std::cout << "\t Strip " << iCC << " value " << (*signalCoupling)[iCC] << std::endl;
      //

      int low = std::max(0,i-nSignalCoupling+1);
      int high = std::min(numStrips-1,i+nSignalCoupling-1);
      if((int)minCha>low) minCha=low;
      if((int)maxCha<high)  maxCha=high;
      double fact = (totalIntegrationRange/geVperElectron)*
	(*sp).amplitude();
      for (int j = low ; j<=high ; j++) 
	locAmpl[j] += (*signalCoupling)[abs(j-i)]*fact; 
      
      /*
	for (int j = -nSignalCoupling+1 ; j<=nSignalCoupling-1 ; j++) {
	if (i+j >= 0 && i+j < numStrips ) {
	locAmpl[i+j] += signalCoupling[abs(j)]* 
	(totalIntegrationRange/geVperElectron)*
	(*sp).amplitude();
	// update counters edges
	if((int)minCha>(i+j))
	minCha=(i+j);
	if((int)maxCha<(i+j))
	maxCha=(i+j);
	//
	}
	}
       */
    } //loop on i
  } //loop on sp
  
}
