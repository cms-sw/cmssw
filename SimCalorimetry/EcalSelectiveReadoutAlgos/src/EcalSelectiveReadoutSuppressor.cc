#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include<limits>


using namespace boost;
//TODO
int XSIZE = 300;
int YSIZE = 300;

EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor()
{
  initTowerThresholds(2.5, 5.0, 2, 2);
  initCellThresholds(0.09, 0.45);
}

EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params) 
{
  initTowerThresholds( params.getParameter<double>("srpLowTowerThreshold"), 
               params.getParameter<double>("srpHighTowerThreshold"),
               params.getParameter<int>("deltaEta"),
               params.getParameter<int>("deltaPhi") );
  initCellThresholds(params.getParameter<double>("srpBarrelLowInterest"),
                     params.getParameter<double>("srpEndcapLowInterest"));
}


void EcalSelectiveReadoutSuppressor::setTriggerMap(const EcalTrigTowerConstituentsMap * map) 
{
  ecalSelectiveReadout->setTriggerMap(map);
}


void EcalSelectiveReadoutSuppressor::setGeometry(const CaloGeometry * caloGeometry) 
{
  ecalSelectiveReadout->setGeometry(caloGeometry);
}


void EcalSelectiveReadoutSuppressor::initTowerThresholds(double lowTowerThreshold, double highTowerThreshold,
                                                 int deltaEta, int deltaPhi) 
{
  std::vector<double> srpThr(2);
  srpThr[0]= lowTowerThreshold;
  srpThr[1]= highTowerThreshold;
  ecalSelectiveReadout = new EcalSelectiveReadout(srpThr,deltaEta,deltaPhi);
}


void EcalSelectiveReadoutSuppressor::initCellThresholds(double barrelLowInterest, double endcapLowInterest)
{ 
  float MINUS_INFINITY = -std::numeric_limits<float>::max();
  zsThreshold[BARREL][0] = barrelLowInterest;
  zsThreshold[BARREL][1] = MINUS_INFINITY;
  zsThreshold[ENDCAP][0] = endcapLowInterest;
  zsThreshold[ENDCAP][1] = MINUS_INFINITY;
  
  zsThreshold[BARREL][2]=MINUS_INFINITY;
  zsThreshold[BARREL][3]=MINUS_INFINITY;
  zsThreshold[ENDCAP][2]=MINUS_INFINITY;
  zsThreshold[ENDCAP][3]=MINUS_INFINITY;
}


double EcalSelectiveReadoutSuppressor::threshold(const EBDetId & detId) const {
  int interestLevel = ecalSelectiveReadout->getCrystalInterest(detId);
  return zsThreshold[0][interestLevel];
}


double EcalSelectiveReadoutSuppressor::threshold(const EEDetId & detId) const {
  int interestLevel = ecalSelectiveReadout->getCrystalInterest(detId);
  return zsThreshold[1][interestLevel];
}


double EcalSelectiveReadoutSuppressor::energy(const EBDataFrame & frame) const {
  //@@ TODO
  double e = 0.;
  return e;
}


double EcalSelectiveReadoutSuppressor::energy(const EEDataFrame & frame) const {
  //@@ TODO 
  double e = 0.;
  return e;
}


double EcalSelectiveReadoutSuppressor::Et(const EcalTriggerPrimitiveDigi & trigPrim) const {
  //TODO make this realistic!
  return trigPrim[5].compressedEt();
}


double Et(const EcalTriggerPrimitiveDigi & trigPrim) {
  //TODO make this realistic!
  return trigPrim[5].compressedEt();
}


void EcalSelectiveReadoutSuppressor::run(
           const EcalTrigPrimDigiCollection & trigPrims,
           EBDigiCollection & barrelDigis,
           EEDigiCollection & endcapDigis){
  EBDigiCollection selectedBarrelDigis;
  EEDigiCollection selectedEndcapDigis;

  run(trigPrims, barrelDigis, endcapDigis,
      selectedBarrelDigis, selectedEndcapDigis);
  
//replaces the input with the suppressed version
  barrelDigis.swap(selectedBarrelDigis);
  endcapDigis.swap(selectedEndcapDigis);  
}


void
EcalSelectiveReadoutSuppressor::run(const EcalTrigPrimDigiCollection & trigPrims,
				    const EBDigiCollection & barrelDigis,
				    const EEDigiCollection & endcapDigis,
				    EBDigiCollection & selectedBarrelDigis,
				    EEDigiCollection & selectedEndcapDigis)
{
  setTriggerTowers(trigPrims);
  ecalSelectiveReadout->runSelectiveReadout0(triggerEt);

  // do barrel first
  for(EBDigiCollection::const_iterator digiItr = barrelDigis.begin();
      digiItr != barrelDigis.end(); ++digiItr)
  {
    if( energy(*digiItr) >= threshold(digiItr->id()) ) {
      selectedBarrelDigis.push_back(*digiItr);
    } 
  }

  // and endcaps
  for(EEDigiCollection::const_iterator digiItr = endcapDigis.begin();
      digiItr != endcapDigis.end(); ++digiItr)
  {
    if( energy(*digiItr) >= threshold(digiItr->id()) ) {
      selectedEndcapDigis.push_back(*digiItr);
    }
  }
}



void EcalSelectiveReadoutSuppressor::setTriggerTowers(const EcalTrigPrimDigiCollection & trigPrims){
  
    for(size_t iEta = 0; iEta < nTriggerTowersInEta; ++iEta){
      for(size_t iPhi = 0; iPhi < nTriggerTowersInPhi; ++iPhi){
        triggerEt[iEta][iPhi] = 0.;
        triggerEt[iEta][iPhi] = 0.;
      }
    }

    int iTrigPrim = 0;
    for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims.begin();
        trigPrim != trigPrims.end(); ++trigPrim)
    {
      float et = Et(*trigPrim); //or etWithoutBXID() ???
       // we want the indexing to go from zero.
      unsigned int eta = trigPrim->id().ieta() + nTriggerTowersInEta/2 - 1;
      unsigned int phi = trigPrim->id().iphi() - 1;
      assert(eta<nTriggerTowersInEta);
      assert(phi<nTriggerTowersInPhi);

//TODO is this still true?
/*
      if(eta>1 || eta < 54){//this detector eta-section part is divided in 72 phi bins
        triggerEt[eta][phi] = et;
      } else{//this detector eta-section is divided in only 36 phi bins
	//For this eta regions,
	//current tower eta numbering scheme is inconsistent. For geometry
	//version 133:
	//- TT are numbered from 0 to 72 for 36 bins
	//- some TT have an even index, some an odd index
	//For geometry version 125, there are 72 phi bins.
	//The code below should handle both geometry definition.
	//If there are 72 input trigger primitives for each inner eta-ring,
	//then the average of the trigger primitive of the two pseudo-TT of
	//a pair (nEta, nEta+1) is taken as Et of both pseudo TTs.
	//If there are only 36 input TTs for each inner eta ring, then half
	//of the present primitive of a pseudo TT pair is used as Et of both
	//pseudo TTs.
      
	//Gets the even index of the pseudo-TT pair this TT belong to:
	int phiEvenIndex = (phi/2)*2; //integer arithmetic
      
	//divides the TT into 2 phi bins in order to match with 72 phi-bins SRP
	//scheme or average the Et on the two pseudo TTs if the TT is already
	//divided into two trigger primitives.
	triggerEt[eta][phiEvenIndex][iz] += et/2.;
	triggerEt[eta][phiEvenIndex+1][iz] += et/2.;
      }
*/
      triggerEt[eta][phi] += et;
      ++iTrigPrim;
    }
    //checks trigger primitive count:
    // with geom 133 we must have 4 eta bins divided in 36 trigger towers
    //        and 52 eta bins divided in 72 trigger towers:
    //  
    const int expectedTrigPrimCount133 = 36*4 + 72*52;
  
    // with geom 125 we must have 56 eta bins divided in 72 trigger towers
    const int expectedTrigPrimCount125 = 72*56;  
  
    if(iTrigPrim!=expectedTrigPrimCount133
       && iTrigPrim!=expectedTrigPrimCount125 ){//wrong number of trigger primitives
      std::cout << "Error. Number of trigger primitive is wrong ("
		<< iTrigPrim << " instead of " << expectedTrigPrimCount125
		<< " or " << expectedTrigPrimCount133
		<< "). It can happened if they were erroneously zero "
	"suppressed (see bug reports #7069 and #7229). Running with trigger "
	"primitive reconstruction forced may "
	"solve the problem." << std::endl;
    }
  }
