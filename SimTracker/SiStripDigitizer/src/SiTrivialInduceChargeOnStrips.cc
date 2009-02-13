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
  }else{
    // TIB
    signalCoupling_TIB = conf_.getParameter<std::vector<double> >("CouplingCostantDecTIB");
    // TID
    signalCoupling_TID = conf_.getParameter<std::vector<double> >("CouplingCostantDecTID");
    // TOB
    signalCoupling_TOB = conf_.getParameter<std::vector<double> >("CouplingCostantDecTOB");
    // TEC
    signalCoupling_TEC = conf_.getParameter<std::vector<double> >("CouplingCostantDecTEC");
  }
  clusterWidth = 3.; 
  geVperElectron = g;
}
void SiTrivialInduceChargeOnStrips::induce(SiChargeCollectionDrifter::collection_type _collection_points, const StripGeomDetUnit& det, 
					   std::vector<double>& locAmpl, unsigned int& minCha, unsigned int& maxCha){
  // Variables and topology
  const StripTopology& t = dynamic_cast<const StripTopology&>(det.specificTopology()); 
  int numStrips = t.nstrips();
  int stripLeft, stripRight;
  double upperBound, lowerBound;
  
  minCha=locAmpl.size();
  maxCha=0;

  // Variables defined here to speedup the loop
  float chargePosition, localPitch, chargeSpread;
  double totalIntegrationRange, fact;
  int low, high;

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

  for (SiChargeCollectionDrifter::collection_type::const_iterator sp=_collection_points.begin();  sp != _collection_points.end(); sp++ ){
    chargePosition = t.strip((*sp).position());  // charge in strip coord
    localPitch = t.localPitch((*sp).position()); // local strip pitch 
    chargeSpread = (*sp).sigma()/localPitch ;    // sigma in strip coord
    
    // Define strips intervals along x: check edge condition
    stripLeft  = int( chargePosition-clusterWidth*chargeSpread);
    stripRight = int( chargePosition+clusterWidth*chargeSpread);
    stripLeft  = (0<stripLeft ? stripLeft : 0);
    stripRight = (numStrips >stripRight ? stripRight : numStrips-1);
   
    for (int i=stripLeft; i<=stripRight; i++){
      // Definition of the integration borns
      lowerBound = (i == 0) ? 0. : 1. - ROOT::Math::normal_cdf_c(i,chargeSpread,chargePosition);
      upperBound = (i == numStrips-1) ? 1. : 1. - ROOT::Math::normal_cdf_c(i+1,chargeSpread,chargePosition);
      totalIntegrationRange = upperBound - lowerBound;
      //calculate signal on strips including capacitive coupling
      low = std::max(0,i-nSignalCoupling+1);
      high = std::min(numStrips-1,i+nSignalCoupling-1);
      if((int)minCha>low) minCha=low;
      if((int)maxCha<high)  maxCha=high;
      fact = (totalIntegrationRange/geVperElectron)*(*sp).amplitude();
      for (int j = low ; j<=high ; j++) 
	locAmpl[j] += (*signalCoupling)[abs(j-i)]*fact; 
    } //loop on i
  } //loop on sp
}
