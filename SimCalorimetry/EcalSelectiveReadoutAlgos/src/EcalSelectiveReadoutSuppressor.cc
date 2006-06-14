#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include<limits>

using namespace boost;
using namespace std;

EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params):
  firstFIRSample(params.getParameter<int>("ecalDccZs1stSample")),
  nFIRTaps(6),
  weights(params.getParameter<vector<double> >("dccNormalizedWeights"))
{
  double adcToGeV = params.getParameter<double>("ebDccAdcToGeV");
  ebMeV2ADC = adcToGeV!=0?1.e3/adcToGeV:0.;
  adcToGeV = params.getParameter<double>("eeDccAdcToGeV");
  eeMeV2ADC = adcToGeV!=0?1.e3/adcToGeV:0.;
  initTowerThresholds( params.getParameter<double>("srpLowTowerThreshold"), 
               params.getParameter<double>("srpHighTowerThreshold"),
               params.getParameter<int>("deltaEta"),
               params.getParameter<int>("deltaPhi") );
  initCellThresholds(params.getParameter<double>("srpBarrelLowInterestChannelZS"),
                     params.getParameter<double>("srpEndcapLowInterestChannelZS"),
		     params.getParameter<double>("srpBarrelHighInterestChannelZS"),
                     params.getParameter<double>("srpEndcapHighInterestChannelZS")
		     );
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


void EcalSelectiveReadoutSuppressor::initCellThresholds(double barrelLowInterest, double endcapLowInterest, double barrelHighInterest, double endcapHighInterest)
{ 
  zsThreshold[BARREL][0] = barrelLowInterest;
  zsThreshold[BARREL][1] = barrelHighInterest;
  zsThreshold[ENDCAP][0] = endcapLowInterest;
  zsThreshold[ENDCAP][1] = endcapHighInterest;
  
  zsThreshold[BARREL][2] = barrelHighInterest;
  zsThreshold[BARREL][3] = barrelHighInterest;
  zsThreshold[ENDCAP][2] = endcapHighInterest;
  zsThreshold[ENDCAP][3] = endcapHighInterest;
}


double EcalSelectiveReadoutSuppressor::threshold(const EBDetId & detId) const {
  int interestLevel = ecalSelectiveReadout->getCrystalInterest(detId);
   return interestLevel!=EcalSelectiveReadout::UNKNOWN?zsThreshold[0][interestLevel]:-numeric_limits<double>::max();
 }


double EcalSelectiveReadoutSuppressor::threshold(const EEDetId & detId) const {
  int interestLevel = ecalSelectiveReadout->getCrystalInterest(detId);
   return interestLevel!=EcalSelectiveReadout::UNKNOWN?zsThreshold[1][interestLevel]:-numeric_limits<double>::max();
}

//This implementation  assumes that int is coded on at least 28-bits,
//which in pratice should be always true.
//T is expectec to be EEDataFrame or EBDataFrame class
template<class T>
bool EcalSelectiveReadoutSuppressor::accept(const T& frame,
					    float threshold){
  //TODO: change ebMeV2ADC to eeMeV2ADC for endcap!!!!
  
  int thr = (int)(threshold * ebMeV2ADC * 4. + .5);
  //treating over- and underflows, threshold is coded on 11+1 signed bits
  if(thr>0x7FF){
    thr = 0x7FF;
  }
  if(thr<-0x7FF){
    thr = -0x7FF;
  }


  //FIR filter weights:
  const vector<int>& w = getFIRWeigths();
  
  //accumulator used to compute weighted sum of samples
  int acc = 0;
  bool gain12saturated = false;
  const int gain12 = 0x01;
  const int lastFIRSample = firstFIRSample + nFIRTaps - 1;
  for(int i=firstFIRSample; i<lastFIRSample; ++i){
    if(i>=0 && i < frame.size()){
      const EcalMGPASample& sample = frame[i];
      if(sample.gainId()!=gain12) gain12saturated = true;
      acc+=sample.adc()*w[i];
    } else{
      //TODO: deals properly logging...
      cout << __FILE__ << ":" << __LINE__ <<
	": Not enough samples in data frame...\n";
    }
  }
  //discards the 8 LSBs
  //(shift operator cannot be used on negative numbers because
  // the result depends on compilator implementation)
  acc = (acc>=0)?(acc >> 8):-(-acc >> 8);
  //ZS passed if weigthed sum acc above ZS threshold or if
  //one sample has a lower gain than gain 12 (that is gain 12 output
  //is saturated)
  return (acc>=threshold || gain12saturated);
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
  setTtFlags(trigPrims);
  
  ecalSelectiveReadout->runSelectiveReadout0(ttFlags);

  // do barrel first
  for(EBDigiCollection::const_iterator digiItr = barrelDigis.begin();
      digiItr != barrelDigis.end(); ++digiItr){
    if(accept(*digiItr, threshold(digiItr->id()))){
      selectedBarrelDigis.push_back(*digiItr);
    } 
  }

  // and endcaps
  for(EEDigiCollection::const_iterator digiItr = endcapDigis.begin();
      digiItr != endcapDigis.end(); ++digiItr){
    if(accept(*digiItr, threshold(digiItr->id()))){
      selectedEndcapDigis.push_back(*digiItr);
    }
  }
}


void EcalSelectiveReadoutSuppressor::setTtFlags(const EcalTrigPrimDigiCollection & trigPrims){
  for(size_t iEta = 0; iEta < nTriggerTowersInEta; ++iEta){
    for(size_t iPhi = 0; iPhi < nTriggerTowersInPhi; ++iPhi){
      ttFlags[iEta][iPhi] = EcalSelectiveReadout::TTF_FORCED_RO_OTHER1;
    }
  }
  for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims.begin();
      trigPrim != trigPrims.end(); ++trigPrim){
    int iEta0 =  trigPrim->id().ieta();
    unsigned int iEta;
    if(iEta0<0){ //z- half ECAL: transforming ranges -28;-1 => 0;27
      iEta = iEta0 + nTriggerTowersInEta/2;
    } else{ //z+ halfECAL: transforming ranges 1;28 => 28;55
      iEta = iEta0 + nTriggerTowersInEta/2 - 1;
    }
    
    unsigned int iPhi = trigPrim->id().iphi() - 1;
    const int iTPSample = 4;
    if(trigPrim->size()>=iTPSample){
      ttFlags[iEta][iPhi] = (EcalSelectiveReadout::ttFlag_t)(*trigPrim)[iTPSample].ttFlag();
    }
  }
}


// void EcalSelectiveReadoutSuppressor::setTriggerTowers(const EcalTrigPrimDigiCollection & trigPrims){
  
//     for(size_t iEta = 0; iEta < nTriggerTowersInEta; ++iEta){
//       for(size_t iPhi = 0; iPhi < nTriggerTowersInPhi; ++iPhi){
//         triggerEt[iEta][iPhi] = 0.;
//         triggerEt[iEta][iPhi] = 0.;
//       }
//     }

//     int iTrigPrim = 0;
//     for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims.begin();
//         trigPrim != trigPrims.end(); ++trigPrim)
//     {
//       float et = Et(*trigPrim); //or etWithoutBXID() ???
//        // we want the indexing to go from zero.
//       int eta0 =  trigPrim->id().ieta();
//       unsigned int eta;
//       if(eta0<0){ //z- half ECAL: transforming ranges -28;-1 => 0;27
// 	eta = eta0 + nTriggerTowersInEta/2;
//       } else{ //z+ halfECAL: transforming ranges 1;28 => 28;55
// 	eta = eta0 + nTriggerTowersInEta/2 - 1;
//       }
//       unsigned int phi = trigPrim->id().iphi() - 1;
//       assert(eta<nTriggerTowersInEta);
//       assert(phi<nTriggerTowersInPhi);

// //TODO is this still true?
// /*
//       if(eta>1 || eta < 54){//this detector eta-section part is divided in 72 phi bins
//         triggerEt[eta][phi] = et;
//       } else{//this detector eta-section is divided in only 36 phi bins
// 	//For this eta regions,
// 	//current tower eta numbering scheme is inconsistent. For geometry
// 	//version 133:
// 	//- TT are numbered from 0 to 72 for 36 bins
// 	//- some TT have an even index, some an odd index
// 	//For geometry version 125, there are 72 phi bins.
// 	//The code below should handle both geometry definition.
// 	//If there are 72 input trigger primitives for each inner eta-ring,
// 	//then the average of the trigger primitive of the two pseudo-TT of
// 	//a pair (nEta, nEta+1) is taken as Et of both pseudo TTs.
// 	//If there are only 36 input TTs for each inner eta ring, then half
// 	//of the present primitive of a pseudo TT pair is used as Et of both
// 	//pseudo TTs.
      
// 	//Gets the even index of the pseudo-TT pair this TT belong to:
// 	int phiEvenIndex = (phi/2)*2; //integer arithmetic
      
// 	//divides the TT into 2 phi bins in order to match with 72 phi-bins SRP
// 	//scheme or average the Et on the two pseudo TTs if the TT is already
// 	//divided into two trigger primitives.
// 	triggerEt[eta][phiEvenIndex][iz] += et/2.;
// 	triggerEt[eta][phiEvenIndex+1][iz] += et/2.;
//       }
// */
//       triggerEt[eta][phi] += et;
//       ++iTrigPrim;
//     }
//     //checks trigger primitive count:
//     // with geom 133 we must have 4 eta bins divided in 36 trigger towers
//     //        and 52 eta bins divided in 72 trigger towers:
//     //  
//     const int expectedTrigPrimCount133 = 36*4 + 72*52;
  
//     // with geom 125 we must have 56 eta bins divided in 72 trigger towers
//     const int expectedTrigPrimCount125 = 72*56;  
  
//     if(iTrigPrim!=expectedTrigPrimCount133
//        && iTrigPrim!=expectedTrigPrimCount125 ){//wrong number of trigger primitives
//       std::cout << "Error. Number of trigger primitive is wrong ("
// 		<< iTrigPrim << " instead of " << expectedTrigPrimCount125
// 		<< " or " << expectedTrigPrimCount133
// 		<< "). It can happened if they were erroneously zero "
// 	"suppressed (see bug reports #7069 and #7229). Running with trigger "
// 	"primitive reconstruction forced may "
// 	"solve the problem." << std::endl;
//     }
//   }

vector<int> EcalSelectiveReadoutSuppressor::getFIRWeigths(){
  if(firWeights.size()==0){
    firWeights = vector<int>(nFIRTaps, 0); //default weight: 0;
    const static int maxWeight = 0xEFF; //weights coded on 11+1 signed bits
    for(unsigned i=0; i < weights.size(); ++i){ 
      firWeights[i] = (int)(weights[i] * (1<<10) + 0.5);
      if(abs(firWeights[i])>maxWeight){//overflow
	firWeights[i] = firWeights[i]<0?-maxWeight:maxWeight;
      }
    }
  }
  return firWeights;
}
