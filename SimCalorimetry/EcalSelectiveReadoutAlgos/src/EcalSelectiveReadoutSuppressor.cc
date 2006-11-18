#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

//exceptions:
#include "FWCore/Utilities/interface/Exception.h"

#include <limits>
#include <cmath>

using namespace boost;
using namespace std;

const int EcalSelectiveReadoutSuppressor::nFIRTaps = 6;

#include <iostream>
EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params)//:
  //firstFIRSample(params.getParameter<int>("ecalDccZs1stSample")),
  //weights(params.getParameter<vector<double> >("dccNormalizedWeights"))
{
 firstFIRSample = params.getParameter<int>("ecalDccZs1stSample");
 weights = params.getParameter<vector<double> >("dccNormalizedWeights");
  double adcToGeV = params.getParameter<double>("ebDccAdcToGeV");
  ebGeV2ADC = adcToGeV!=0?1./adcToGeV:0.;
  adcToGeV = params.getParameter<double>("eeDccAdcToGeV");
  eeGeV2ADC = adcToGeV!=0?1./adcToGeV:0.;
  initTowerThresholds( params.getParameter<double>("srpLowTowerThreshold"), 
               params.getParameter<double>("srpHighTowerThreshold"),
               params.getParameter<int>("deltaEta"),
               params.getParameter<int>("deltaPhi") );
  initCellThresholds(params.getParameter<double>("srpBarrelLowInterestChannelZS"),
                     params.getParameter<double>("srpEndcapLowInterestChannelZS"),
		     params.getParameter<double>("srpBarrelHighInterestChannelZS"),
                     params.getParameter<double>("srpEndcapHighInterestChannelZS")
		     );
  trigPrimBypass_ = params.getParameter<bool>("trigPrimBypass");
  trigPrimBypassWithPeakFinder_
    = params.getParameter<bool>("trigPrimBypassWithPeakFinder");
  trigPrimBypassLTH_ = params.getParameter<double>("trigPrimBypassLTH");
  trigPrimBypassHTH_ = params.getParameter<double>("trigPrimBypassHTH");
  if(trigPrimBypass_){
    edm::LogWarning("Digitization") << "Beware a simplified trigger primitive "
       "computation is used for the ECAL selective readout";
   }
}


void EcalSelectiveReadoutSuppressor::setTriggerMap(const EcalTrigTowerConstituentsMap * map) 
{
  theTriggerMap = map;
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
template<class T>
bool EcalSelectiveReadoutSuppressor::accept(const T& frame,
					    float threshold){
  double eGeV2ADC;
  switch(frame.id().subdetId()){
  case EcalBarrel:
    eGeV2ADC = ebGeV2ADC;
    break;
  case EcalEndcap:
    eGeV2ADC = eeGeV2ADC;
    break;
  default:
    throw cms::Exception("EcalSelectiveReadoutSuppressor: unexpected subdetector id in a dataframe. Only EB and EE data frame are expected.");
  }
  
  double thr_ = threshold * eGeV2ADC * 4.;
  //treating over- and underflows, threshold is coded on 11+1 signed bits
  //an underflow threshold is considered here as if NoRO DCC switch is on
  //an overflow threshold is considered here as if ForcedRO DCC switch in on
  //Beware that conparison must be done on a double type, because conversion
  //cast to an int of a double higher than MAX_INT is undefined.
  int thr;
  if(thr_>=0x7FF+.5){
    thr = numeric_limits<int>::max();
  } else if(thr_<=-0x7FF-.5){
    thr = -numeric_limits<int>::min();
  } else{
    thr = lround(thr_);
  }
  
  //FIR filter weights:
  const vector<int>& w = getFIRWeigths();
  
  //accumulator used to compute weighted sum of samples
  int acc = 0;
  bool gain12saturated = false;
  const int gain12 = 0x01; 
  const int lastFIRSample = firstFIRSample + nFIRTaps - 1;
  LogDebug("DccFir") << "DCC FIR operation: ";
  for(int i=firstFIRSample-1; i<lastFIRSample; ++i){
    if(i>=0 && i < frame.size()){
      const EcalMGPASample& sample = frame[i];
      if(sample.gainId()!=gain12) gain12saturated = true;
      LogTrace("DccFir")  << (i>=firstFIRSample?"+":"") << sample.adc()
			  << "*(" << w[i] << ")";
      acc+=sample.adc()*w[i];
    } else{
      edm::LogWarning("DccFir") << __FILE__ << ":" << __LINE__ <<
	": Not enough samples in data frame or 'ecalDccZs1stSample' module "
	"parameter is not valid...";
    }
  }
  LogTrace("DccFir") << "\n";
  //discards the 8 LSBs
  //(shift operator cannot be used on negative numbers because
  // the result depends on compilator implementation)
  acc = (acc>=0)?(acc >> 8):-(-acc >> 8);
  //ZS passed if weigthed sum acc above ZS threshold or if
  //one sample has a lower gain than gain 12 (that is gain 12 output
  //is saturated)

  const bool result = acc>=thr || gain12saturated;
  
  LogTrace("DccFir") << "acc: " << acc << "\n"
		     << "threshold: " << thr << " (" << threshold << "GeV)\n"
		     << "saturated: " << (gain12saturated?"yes":"no") << "\n"
		     << "ZS passed: " << (result?"yes":"no") << "\n";
  
  return result;
}


int EcalSelectiveReadoutSuppressor::accumulate(const EcalDataFrame & frame,
                                               bool & gain12saturated)
{
  //FIR filter weights:
  const vector<int>& w = getFIRWeigths();

  int acc = 0;
  gain12saturated = false;
  const int gain12 = 0x01;
  const int lastFIRSample = firstFIRSample + nFIRTaps - 1;
  LogDebug("DccFir") << "DCC FIR operation: ";
  for(int i=firstFIRSample-1; i<lastFIRSample; ++i){
    if(i>=0 && i < frame.size()){
      const EcalMGPASample& sample = frame[i];
      if(sample.gainId()!=gain12) gain12saturated = true;
      LogTrace("DccFir")  << (i>=firstFIRSample?"+":"") << sample.adc()
        << "*(" << w[i] << ")";
      acc+=sample.adc()*w[i];
    } else{
      edm::LogWarning("DccFir") << __FILE__ << ":" << __LINE__ <<
  ": Not enough samples in data frame or 'ecalDccZs1stSample' module "
  "parameter is not valid...";
    }
  }
  return acc;
}


double EcalSelectiveReadoutSuppressor::energy(const EcalDataFrame & frame)
{
  bool gain12saturated;
  double acc = accumulate(frame, gain12saturated);
  double adc2GeV;
  switch(frame.id().subdetId()){
  case EcalBarrel:
    adc2GeV = 1./ebGeV2ADC;
    break;
  case EcalEndcap:
    adc2GeV = 1./eeGeV2ADC;
    break;
  }
  acc *= (adc2GeV / (1<<10));
  return acc;
}


void EcalSelectiveReadoutSuppressor::run(const edm::EventSetup& eventSetup,   
					 const EcalTrigPrimDigiCollection & trigPrims,
					 EBDigiCollection & barrelDigis,
					 EEDigiCollection & endcapDigis){
  EBDigiCollection selectedBarrelDigis;
  EEDigiCollection selectedEndcapDigis;

  run(eventSetup, trigPrims, barrelDigis, endcapDigis,
      selectedBarrelDigis, selectedEndcapDigis);
  
//replaces the input with the suppressed version
  barrelDigis.swap(selectedBarrelDigis);
  endcapDigis.swap(selectedEndcapDigis);  
}


void
EcalSelectiveReadoutSuppressor::run(const edm::EventSetup& eventSetup,
				    const EcalTrigPrimDigiCollection & trigPrims,
				    const EBDigiCollection & barrelDigis,
				    const EEDigiCollection & endcapDigis,
				    EBDigiCollection & selectedBarrelDigis,
				    EEDigiCollection & selectedEndcapDigis)
{
  if(!trigPrimBypass_){//normal mode
    setTtFlags(trigPrims);
  } else{//debug mode, run w/o TP digis
    setTtFlags(eventSetup, barrelDigis, endcapDigis);
  }
  
  ecalSelectiveReadout->runSelectiveReadout0(ttFlags);

  // do barrel first
  for(EBDigiCollection::const_iterator digiItr = barrelDigis.begin();
      digiItr != barrelDigis.end(); ++digiItr){
    //TO DO: remove EBDetId conversion once EBxxxDigi::id() return type fixed 
    if(accept(*digiItr, threshold(EBDetId(digiItr->id())))){
      selectedBarrelDigis.push_back(*digiItr);
    } 
  }
  
  // and endcaps
  for(EEDigiCollection::const_iterator digiItr = endcapDigis.begin();
      digiItr != endcapDigis.end(); ++digiItr){
    //TO DO: remove EEDetId conversion once EBxxxDigi::id() return type fixed 
    if(accept(*digiItr, threshold(EEDetId(digiItr->id())))){
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

    //TODO: code below must be change to use
    //EcalTriggerPrimitiveDigi::ttFlags() method, once available:
    int iTPSample;
    if(trigPrim->size()>8){//version before nb of Tp data samples was changed
      iTPSample = 5;
    } else{
      iTPSample = 2;
    }
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

vector<int> EcalSelectiveReadoutSuppressor::getFIRWeigths() {
  if(firWeights.size()==0){
    firWeights = vector<int>(nFIRTaps, 0); //default weight: 0;
    const static int maxWeight = 0xEFF; //weights coded on 11+1 signed bits
    for(unsigned i=0; i < min((unsigned)nFIRTaps,weights.size()); ++i){ 
      firWeights[i] = lround(weights[i] * (1<<10));
      if(abs(firWeights[i])>maxWeight){//overflow
	firWeights[i] = firWeights[i]<0?-maxWeight:maxWeight;
      }
    }
  }
  return firWeights;
}

void
EcalSelectiveReadoutSuppressor::setTtFlags(const edm::EventSetup& es,
					   const EBDigiCollection& ebDigis,
					   const EEDigiCollection& eeDigis){
  double trigPrim[nTriggerTowersInEta][nTriggerTowersInPhi];

  //ecal geometry:
  static const CaloSubdetectorGeometry* eeGeometry = 0;
  static const CaloSubdetectorGeometry* ebGeometry = 0;
  if(eeGeometry==0 || ebGeometry==0){
    edm::ESHandle<CaloGeometry> geoHandle;
    es.get<IdealGeometryRecord>().get(geoHandle);
    eeGeometry
      = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    ebGeometry
      = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  }

  //init trigPrim array:
  bzero(trigPrim, sizeof(trigPrim));
	
  for(EBDigiCollection::const_iterator it = ebDigis.begin();
      it != ebDigis.end(); ++it){
    const EBDataFrame& frame = *it;
    const EcalTrigTowerDetId& ttId = theTriggerMap->towerOf(frame.id());
//      edm:::LogDebug("TT") << __FILE__ << ":" << __LINE__ << ": " 
//  	 <<  ((EBDetId&)frame.id()).ieta()
//  	 << "," << ((EBDetId&)frame.id()).iphi()
//  	 << " -> " << ttId.ieta() << "," << ttId.iphi() << "\n";
    const int iTTEta0 = iTTEta2cIndex(ttId.ieta());
    const int iTTPhi0 = iTTPhi2cIndex(ttId.iphi());
    double theta = ebGeometry->getGeometry(frame.id())->getPosition().theta();
    double e = frame2Energy(frame);
    if(!trigPrimBypassWithPeakFinder_
       || ((frame2Energy(frame,-1) < e) && (frame2Energy(frame, 1) < e))){
      trigPrim[iTTEta0][iTTPhi0] += e*sin(theta);
    }
  }

  for(EEDigiCollection::const_iterator it = eeDigis.begin();
      it != eeDigis.end(); ++it){
    const EEDataFrame& frame = *it;
    const EcalTrigTowerDetId& ttId = theTriggerMap->towerOf(frame.id());
    const int iTTEta0 = iTTEta2cIndex(ttId.ieta());
    const int iTTPhi0 = iTTPhi2cIndex(ttId.iphi());
//     cout << __FILE__ << ":" << __LINE__ << ": EE xtal->TT "
// 	 <<  ((EEDetId&)frame.id()).ix()
// 	 << "," << ((EEDetId&)frame.id()).iy()
// 	 << " -> " << ttId.ieta() << "," << ttId.iphi() << "\n";
    double theta = eeGeometry->getGeometry(frame.id())->getPosition().theta();
    double e = frame2Energy(frame);
    if(!trigPrimBypassWithPeakFinder_
       || ((frame2Energy(frame,-1) < e) && (frame2Energy(frame, 1) < e))){
      trigPrim[iTTEta0][iTTPhi0] += e*sin(theta);
    }
  }

  //dealing with pseudo-TT in two inner EE eta-ring:
  int innerTTEtas[] = {0, 1, 54, 55};
  for(unsigned iRing = 0; iRing < sizeof(innerTTEtas)/sizeof(innerTTEtas[0]);
      ++iRing){
    int iTTEta0 = innerTTEtas[iRing];
    //this detector eta-section is divided in only 36 phi bins
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

    for(unsigned iTTPhi0 = 0; iTTPhi0 < nTriggerTowersInPhi-1; iTTPhi0 += 2){
      double et = .5*(trigPrim[iTTEta0][iTTPhi0]
		      +trigPrim[iTTEta0][iTTPhi0+1]);
      //divides the TT into 2 phi bins in order to match with 72 phi-bins SRP
      //scheme or average the Et on the two pseudo TTs if the TT is already
      //divided into two trigger primitives.
      trigPrim[iTTEta0][iTTPhi0] = et;
      trigPrim[iTTEta0][iTTPhi0+1] = et;
    }
  }
    
  for(unsigned iTTEta0 = 0; iTTEta0 < nTriggerTowersInEta; ++iTTEta0){
    for(unsigned iTTPhi0 = 0; iTTPhi0 < nTriggerTowersInPhi; ++iTTPhi0){
      if(trigPrim[iTTEta0][iTTPhi0] > trigPrimBypassHTH_){
	ttFlags[iTTEta0][iTTPhi0] = EcalSelectiveReadout::TTF_HIGH_INTEREST;
      } else if(trigPrim[iTTEta0][iTTPhi0] > trigPrimBypassLTH_){
	ttFlags[iTTEta0][iTTPhi0] = EcalSelectiveReadout::TTF_MID_INTEREST;
      } else{
	ttFlags[iTTEta0][iTTPhi0] = EcalSelectiveReadout::TTF_LOW_INTEREST;
      }
      
      // cout /*LogDebug("TT")*/
      // 	<< "ttFlags[" << iTTEta0 << "][" << iTTPhi0 << "] = "
      // 	<< ttFlags[iTTEta0][iTTPhi0] << "\n";
    }
  }
}

template<class T>
double EcalSelectiveReadoutSuppressor::frame2Energy(const T& frame,
						    int offset) const{
  //we have to start by 0 in order to handle offset=-1
  //(however Fenix FIR has AFAK only 5 taps)
  double weights[] = {0., -1/3., -1/3., -1/3., 0., 1.};   

  double adc2GeV = 0.;
  if(typeid(frame) == typeid(EBDataFrame)){
    adc2GeV = 0.035;
  } else if(typeid(frame) == typeid(EEDataFrame)){
    adc2GeV = 0.060;
  } else{ //T is an invalid type!
    //TODO: replace message by a cms exception
    cerr << "Severe error. "
	 << __FILE__ << ":" << __LINE__ << ": "
	 << "this is a bug. Please report it.\n";
  }
    
  double acc = 0;

  const int n = min<int>(frame.size(), sizeof(weights)/sizeof(weights[0]));

  double gainInv[] = {0., 1., 6., 12}; //first elt not used.


  //cout << __PRETTY_FUNCTION__ << ": ";
  for(int i=offset; i < n; ++i){
    int iframe = i + offset;
    if(iframe>=0 && iframe<frame.size()){
      acc += weights[i]*frame[iframe].adc()
	*gainInv[frame[iframe].gainId()]*adc2GeV;
      //cout << (iframe>offset?"+":"")
      //     << frame[iframe].adc() << "*" << gainInv[frame[iframe].gainId()]
      //     << "*" << adc2GeV << "*(" << weights[i] << ")";
    }
  }
  //cout << "\n";
  return acc;
}
