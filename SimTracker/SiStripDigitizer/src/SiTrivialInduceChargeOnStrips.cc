#include "SimTracker/SiStripDigitizer/interface/SiTrivialInduceChargeOnStrips.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <gsl/gsl_sf_erf.h>
#include<iostream>

SiTrivialInduceChargeOnStrips::SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g):conf_(conf)
{
  peak = conf_.getParameter<bool>("APVpeakmode");
  if(peak){
    coupling_costant = conf_.getParameter<std::vector<double> >("CouplingCostantPeak");
  }else{
    coupling_costant = conf_.getParameter<std::vector<double> >("CouplingCostantDeco");
  }
  clusterWidth = 3.; 
  geVperElectron = g;
}
void SiTrivialInduceChargeOnStrips::induce(SiChargeCollectionDrifter::collection_type _collection_points, const StripGeomDetUnit& det, 
					   SiPileUpSignals::signal_map_type &hit_signal,SiPileUpSignals::signal_map_type &hit_signal_temp){
  signalCoupling.clear();
  signalCoupling.push_back(coupling_costant[0]);
  signalCoupling.push_back(coupling_costant[1]);

  //  const StripTopology& t = dynamic_cast<const StripTopology&>(det.topology());
  const StripTopology& t = dynamic_cast<const StripTopology&>(det.specificTopology()); // AG

  int numStrips = t.nstrips();
  int stripLeft, stripRight;
  double upperBound, lowerBound;
    
    
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
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePosition)/chargeSpread, &result);
	if (status != 0) std::cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	lowerBound = 1. - result.val;
	//      float       olowerBound = freq_((i-chargePosition)/chargeSpread);
        //      cout <<" UPPERB "<<olowerBound<<" " <<lowerBound<<" " <<lowerBound-olowerBound<<endl;
	
	
      }
      if (i == numStrips-1) upperBound = 1.;
      else {
	gsl_sf_result result;
	int status = gsl_sf_erf_Q_e((i-chargePosition+1)/chargeSpread, &result);
		if (status != 0) std::cerr<<"GaussianTailNoiseGenerator::could not compute gaussian tail probability for the threshold chosen"<<std::endl;
	upperBound = 1. - result.val;
	//      float       oupperBound = freq_((i-chargePosition+1)/chargeSpread);
	//      cout <<" UPPERB "<<oupperBound<<" " <<upperBound<<" " <<upperBound-oupperBound<<endl;
  
      }
       
      double totalIntegrationRange = upperBound - lowerBound;
     
      //calculate signal on strips including capacitive coupling
     
      int nSignalCoupling = signalCoupling.size();

      for (int j = -nSignalCoupling+1 ; j<=nSignalCoupling-1 ; j++) {
	if (i+j >= 0 && i+j < numStrips ) { 
	  hit_signal[i+j] += signalCoupling[abs(j)]* 
	    (totalIntegrationRange/geVperElectron)*
	    (*sp).amplitude();   
	  hit_signal_temp[i+j]=hit_signal[i+j];
	}
      }
    } //loop on i
  } //loop on k
}
