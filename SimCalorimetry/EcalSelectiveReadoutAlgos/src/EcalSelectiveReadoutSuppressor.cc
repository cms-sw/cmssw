#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include<limits>


using namespace boost;
//TODO
int XSIZE = 300;
int YSIZE = 300;

EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor()
  :  tower(extents[nEndcaps][XSIZE][YSIZE][2])
{
  initTowerThresholds(2.5, 5.0, 2, 2);
  initCellThresholds(0.09, 0.45);
}

EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params) 
  :  tower(extents[nEndcaps][XSIZE][YSIZE][2])
{
  initTowerThresholds( params.getParameter<double>("srpLowTowerThreshold"), 
               params.getParameter<double>("srpHighTowerThreshold"),
               params.getParameter<int>("deltaEta"),
               params.getParameter<int>("deltaPhi") );
  initCellThresholds(params.getParameter<double>("srpBarrelLowInterest"),
                     params.getParameter<double>("srpEndcapLowInterest"));
}


void EcalSelectiveReadoutSuppressor::initTowerThresholds(double lowTowerThreshold, double highTowerThreshold,
                                                 int deltaEta, int deltaPhi) 
{
  std::vector<double> srpThr(2);
  srpThr[0]= lowTowerThreshold;
  srpThr[1]= highTowerThreshold;
  ecalSelectiveReadout = new EcalSelectiveReadout(srpThr,tower.data(),deltaEta,deltaPhi);
}


void EcalSelectiveReadoutSuppressor::initCellThresholds(double barrelLowInterest, double endcapLowInterest)
{ 
  float MINUS_INFINITY = -std::numeric_limits<float>::max();
  zsThreshold[0][0] = barrelLowInterest;
  zsThreshold[0][1] = MINUS_INFINITY;
  zsThreshold[1][0] = endcapLowInterest;
  zsThreshold[1][1] = MINUS_INFINITY;
  
  zsThreshold[0][2]=MINUS_INFINITY;
  zsThreshold[0][3]=MINUS_INFINITY;
  zsThreshold[1][2]=MINUS_INFINITY;
  zsThreshold[1][3]=MINUS_INFINITY;
}


double EcalSelectiveReadoutSuppressor::threshold(const EBDetId & detId) const {
  int interestLevel = ecalSelectiveReadout->getBarrelCrystalInterest(
    detId.ieta(), detId.iphi());

  return zsThreshold[0][interestLevel];
}


double EcalSelectiveReadoutSuppressor::threshold(const EEDetId & detId) const {
  int interestLevel = ecalSelectiveReadout->getEndcapCrystalInterest(
     detId.zside(), detId.ix(), detId.iy());

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


void EcalSelectiveReadoutSuppressor::setTriggerTowersMap(const CaloSubdetectorGeometry * endcapGeometry,
                                                         const CaloSubdetectorGeometry * towerGeometry)
//const EcalTriggerTowerMapping * mapping)
{
 
  size_t towerSize = tower.shape()[0] * tower.shape()[1]
                   * tower.shape()[2] * tower.shape()[3];

  for(int* pTower = tower.data();
      pTower < tower.data() + towerSize;
      ++pTower)
  {
    *pTower = -1;
  }

  int nCrystal = 0;
  //loops over all the ECAL crystals and sets the trigger tower map
  std::vector<DetId> crystalIds = endcapGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  for(std::vector<DetId>::const_iterator crystalItr = crystalIds.begin();
      crystalItr != crystalIds.end(); ++crystalItr)
  {
    ++nCrystal;
    EEDetId crystalId(*crystalItr);
    //looks for the trigger tower of this crystal
    //TODO
    EcalTrigTowerDetId towerId; // = theTowerMapping->towerId(crystalId);

    tower_t::subarray<1>::type TT
     = tower[crystalId.zside()][crystalId.ix()][crystalId.iy()]; 
    	
    TT[0] = towerId.ieta();
    TT[1] = towerId.iphi();
  }
} 


double Et(const EcalTriggerPrimitiveDigi & trigPrim) {
  //TODO make this realistic!
  return trigPrim[5].compressedEt();
}


void EcalSelectiveReadoutSuppressor::run(
           const EcalTrigPrimDigiCollection & trigPrims,
           EBDigiCollection & barrelDigis,
           EEDigiCollection & endcapDigis)
{

  setTriggerTowers(trigPrims);
  ecalSelectiveReadout->runSelectiveReadout0(triggerEt);


  // do barrel first
  EBDigiCollection newBarrelDigis;
  for(EBDigiCollection::const_iterator digiItr = barrelDigis.begin();
      digiItr != barrelDigis.end(); ++digiItr)
  {
    if( energy(*digiItr) >= threshold(digiItr->id()) ) {
      newBarrelDigis.push_back(*digiItr);
    } 
  }

  // and endcaps
  EEDigiCollection newEndcapDigis;
  for(EEDigiCollection::const_iterator digiItr = endcapDigis.begin();
      digiItr != endcapDigis.end(); ++digiItr)
  {
    if( energy(*digiItr) >= threshold(digiItr->id()) ) {
      newEndcapDigis.push_back(*digiItr);
    }
  }

  // and replace the input with the suppressed version
  barrelDigis.swap(newBarrelDigis);
  endcapDigis.swap(newEndcapDigis);
}



void EcalSelectiveReadoutSuppressor::setTriggerTowers(const EcalTrigPrimDigiCollection & trigPrims){
  
    for(size_t iEta = 0; iEta < nTriggerTowersInEta; ++iEta){
      for(size_t iPhi = 0; iPhi < nTriggerTowersInPhi; ++iPhi){
        triggerEt[iEta][iPhi] = 0.;
      }
    }

    int iTrigPrim = 0;
    for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims.begin();
        trigPrim != trigPrims.end(); ++trigPrim)
    {
      float et = Et(*trigPrim); //or etWithoutBXID() ???
      unsigned int eta = trigPrim->id().ieta();
      unsigned int phi = trigPrim->id().iphi();
      assert(eta<nTriggerTowersInEta);
      assert(phi<nTriggerTowersInPhi);

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
	triggerEt[eta][phiEvenIndex] += et/2.;
	triggerEt[eta][phiEvenIndex+1] += et/2.;
      }
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
