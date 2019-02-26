#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

//exceptions:
#include "FWCore/Utilities/interface/Exception.h"

#include <limits>
#include <cmath>
#include <iostream>

using namespace boost;
using namespace std;

const int EcalSelectiveReadoutSuppressor::nFIRTaps = 6;

EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params, const EcalSRSettings* settings):
  ttThresOnCompressedEt_(false),
  ievt_(0)
{

  firstFIRSample = settings->ecalDccZs1stSample_[0];
  weights = settings->dccNormalizedWeights_[0];
  symetricZS = settings->symetricZS_[0];
  actions_ = settings->actions_;


  int defTtf = params.getParameter<int>("defaultTtf");
  if(defTtf < 0 || defTtf > 7){
    throw cms::Exception("InvalidParameter") << "Value of EcalSelectiveReadoutProducer module parameter defaultTtf, "
					     << defaultTtf_ << ", is out of the valid range 0..7\n";
  } else{
    defaultTtf_ = (EcalSelectiveReadout::ttFlag_t) defTtf; 
  }

  //online configuration has only 4 actions flags, the 4 'forced' flags being the same with the force
  //bit set to 1. Extends the actions vector for case of online-type configuration:
  if(actions_.size()==4){
     for(int i = 0; i < 4; ++i){
        actions_.push_back(actions_[i] | 0x4);
     }
  }


  bool actionValid = actions_.size()==8;
  for(size_t i = 0; i < actions_.size(); ++i){
    if(actions_[i] < 0 || actions_[i] > 7) actionValid = false;
  }

  if(!actionValid){
    throw cms::Exception("InvalidParameter") << "EcalSelectiveReadoutProducer module parameter 'actions' is "
      "not valid. It must be a vector of 8 integer values comprised between 0 and 7\n";
  }
  
  double adcToGeV = settings->ebDccAdcToGeV_;
  thrUnit[BARREL] = adcToGeV/4.; //unit=1/4th ADC count
  
  adcToGeV = settings->eeDccAdcToGeV_;
  thrUnit[ENDCAP] = adcToGeV/4.; //unit=1/4th ADC count
  ecalSelectiveReadout
    = unique_ptr<EcalSelectiveReadout>(new EcalSelectiveReadout(
							      settings->deltaEta_[0],
							      settings->deltaPhi_[0]));
  const int eb = 0;
  const int ee = 1;
  initCellThresholds(settings->srpLowInterestChannelZS_[eb],
		     settings->srpLowInterestChannelZS_[ee],
		     settings->srpHighInterestChannelZS_[eb],
		     settings->srpHighInterestChannelZS_[ee]
		     );
  trigPrimBypass_ = params.getParameter<bool>("trigPrimBypass");
  trigPrimBypassMode_ = params.getParameter<int>("trigPrimBypassMode");
  trigPrimBypassWithPeakFinder_
    = params.getParameter<bool>("trigPrimBypassWithPeakFinder");
  trigPrimBypassLTH_ = params.getParameter<double>("trigPrimBypassLTH");
  trigPrimBypassHTH_ = params.getParameter<double>("trigPrimBypassHTH");
  if(trigPrimBypass_){
    edm::LogWarning("Digitization") << "Beware a simplified trigger primitive "
      "computation is used for the ECAL selective readout";
    if(trigPrimBypassMode_ !=0 && trigPrimBypassMode_ !=1){
      throw cms::Exception("InvalidParameter")
        << "Invalid value for EcalSelectiveReadoutProducer parameter 'trigPrimBypassMode_'."
        " Valid values are 0 and 1.\n";
    }
    ttThresOnCompressedEt_ = (trigPrimBypassMode_==1);
  }
}


void EcalSelectiveReadoutSuppressor::setTriggerMap(const EcalTrigTowerConstituentsMap * map){
  theTriggerMap = map;
  ecalSelectiveReadout->setTriggerMap(map);
}

void EcalSelectiveReadoutSuppressor::setElecMap(const EcalElectronicsMapping * map){
  ecalSelectiveReadout->setElecMap(map);
}


void EcalSelectiveReadoutSuppressor::setGeometry(const CaloGeometry * caloGeometry) 
{
#ifndef ECALSELECTIVEREADOUT_NOGEOM
   ecalSelectiveReadout->setGeometry(caloGeometry);
#endif
}


void EcalSelectiveReadoutSuppressor::initCellThresholds(double barrelLowInterest,
							double endcapLowInterest,
							double barrelHighInterest,
							double endcapHighInterest){
  //center, neighbour and single RUs are grouped into a single
  //'high interest' group
  int lowInterestThr[2]; //index for BARREL/ENDCAP
  //  int lowInterestSrFlag[2];
  int highInterestThr[2];
  // int highInterestSrFlag[2];
  
  lowInterestThr[BARREL] = internalThreshold(barrelLowInterest, BARREL);
  //  lowInterestSrFlag[BARREL] = thr2Srf(lowInterestThr[BARREL],
  //				      EcalSrFlag::SRF_ZS1);
  
  highInterestThr[BARREL] = internalThreshold(barrelHighInterest, BARREL);
  //  highInterestSrFlag[BARREL] = thr2Srf(highInterestThr[BARREL],
  //				       EcalSrFlag::SRF_ZS2);
  
  lowInterestThr[ENDCAP] = internalThreshold(endcapLowInterest, ENDCAP);
  //lowInterestSrFlag[ENDCAP] = thr2Srf(lowInterestThr[ENDCAP],
  //			      EcalSrFlag::SRF_ZS1);

  highInterestThr[ENDCAP] = internalThreshold(endcapHighInterest, ENDCAP); 
  //  highInterestSrFlag[ENDCAP] = thr2Srf(highInterestThr[ENDCAP],
  //			       EcalSrFlag::SRF_ZS2);

  const int FORCED_MASK = EcalSelectiveReadout::FORCED_MASK;
  
  for(int iSubDet = 0; iSubDet<2; ++iSubDet){
    //low interest
    //zsThreshold[iSubDet][0] = lowInterestThr[iSubDet];
    //srFlags[iSubDet][0] = lowInterestSrFlag[iSubDet];
    //srFlags[iSubDet][0 + FORCED_MASK] = FORCED_MASK | lowInterestSrFlag[iSubDet];

    //single->high interest
    //zsThreshold[iSubDet][1] = highInterestThr[iSubDet];
    //srFlags[iSubDet][1] = highInterestSrFlag[iSubDet];
    //srFlags[iSubDet][1 +  FORCED_MASK] = FORCED_MASK | highInterestSrFlag[iSubDet];

    //neighbour->high interest
    //zsThreshold[iSubDet][2] = highInterestThr[iSubDet];
    //srFlags[iSubDet][2] = highInterestSrFlag[iSubDet];
    //srFlags[iSubDet][2 + FORCED_MASK] = FORCED_MASK | highInterestSrFlag[iSubDet];

    //center->high interest
    //zsThreshold[iSubDet][3] = highInterestThr[iSubDet];
    //srFlags[iSubDet][3] = highInterestSrFlag[iSubDet];
    //srFlags[iSubDet][3 + FORCED_MASK] = FORCED_MASK | highInterestSrFlag[iSubDet];
    for(size_t i = 0; i < 8; ++i){
      srFlags[iSubDet][i] = actions_[i];
      if((actions_[i] & ~FORCED_MASK) == 0) zsThreshold[iSubDet][i] = numeric_limits<int>::max();
      else if((actions_[i] & ~FORCED_MASK) == 1) zsThreshold[iSubDet][i] = lowInterestThr[iSubDet];
      else if((actions_[i] & ~FORCED_MASK) == 2)  zsThreshold[iSubDet][i] = highInterestThr[iSubDet];
      else zsThreshold[iSubDet][i] = numeric_limits<int>::min();
    }

//     for(size_t i = 0; i < 8; ++i){
//       cout << "zsThreshold[" << iSubDet << "]["  << i << "] = " << zsThreshold[iSubDet][i] << endl;
//     }
  }
}

// int EcalSelectiveReadoutSuppressor::thr2Srf(int thr, int zsFlag) const{
//   if(thr==numeric_limits<int>::max()){
//     return EcalSrFlag::SRF_SUPPRESS;
//   }
//   if(thr==numeric_limits<int>::min()){
//     return EcalSrFlag::SRF_FULL;
//   } 
//   return zsFlag;
// }

int EcalSelectiveReadoutSuppressor::internalThreshold(double thresholdInGeV,
						      int iSubDet) const{
  double thr_ = thresholdInGeV / thrUnit[iSubDet];
  //treating over- and underflows, threshold is coded on 11+1 signed bits
  //an underflow threshold is considered here as if NoRO DCC switch is on
  //an overflow threshold is considered here as if ForcedRO DCC switch in on
  //Beware that conparison must be done on a double type, because conversion
  //cast to an int of a double higher than MAX_INT is undefined.
  int thr;
  if(thr_>=0x7FF+.5){
    thr = numeric_limits<int>::max();
  } else if(thr_<=-0x7FF-.5){
    thr = numeric_limits<int>::min();
  } else{
    thr = lround(thr_);
  }
  return thr;
}

//This implementation  assumes that int is coded on at least 28-bits,
//which in pratice should be always true.
bool EcalSelectiveReadoutSuppressor::accept(edm::DataFrame const & frame,
					    int thr){
  //FIR filter weights:
  const vector<int>& w = getFIRWeigths();
  
  //accumulator used to compute weighted sum of samples
  int acc = 0;
  bool gain12saturated = false;
  const int gain12 = 0x01; 
  const int lastFIRSample = firstFIRSample + nFIRTaps - 1;
  //LogDebug("DccFir") << "DCC FIR operation: ";
  int iWeight = 0;
  for(int iSample=firstFIRSample-1;
      iSample<lastFIRSample; ++iSample, ++iWeight){
    if(iSample>=0 && iSample < (int)frame.size()){
      EcalMGPASample sample(frame[iSample]);
      if(sample.gainId()!=gain12) gain12saturated = true;
      //LogTrace("DccFir") << (iSample>=firstFIRSample?"+":"") << sample.adc()
      //		 << "*(" << w[iWeight] << ")";
      acc+=sample.adc()*w[iWeight];
    } else{
      edm::LogWarning("DccFir") << __FILE__ << ":" << __LINE__ <<
	": Not enough samples in data frame or 'ecalDccZs1stSample' module "
	"parameter is not valid...";
    }
  }

  if(symetricZS){//cut on absolute value
    if(acc<0) acc = -acc;
  }
  
  //LogTrace("DccFir") << "\n";
  //discards the 8 LSBs 
  //(result of shift operator on negative numbers depends on compiler
  //implementation, therefore the value is offset to make sure it is positive
  //before performing the bit shift).
  acc = ((acc + (1<<30)) >>8) - (1 <<(30-8));

  //ZS passed if weigthed sum acc above ZS threshold or if
  //one sample has a lower gain than gain 12 (that is gain 12 output
  //is saturated)
  
  const bool result = (acc >= thr) || gain12saturated;
  
  //LogTrace("DccFir") << "acc: " << acc << "\n"
  //                   << "threshold: " << thr << " ("
  //                   << thr*thrUnit[((EcalDataFrame&)frame).id().subdetId()==EcalBarrel?0:1]
  //                   << "GeV)\n"
  //                   << "saturated: " << (gain12saturated?"yes":"no") << "\n"
  //                   << "ZS passed: " << (result?"yes":"no")
  //                   << (symetricZS?" (symetric cut)":"") << "\n";
  
  return result;
}

void EcalSelectiveReadoutSuppressor::run(const edm::EventSetup& eventSetup,   
					 const EcalTrigPrimDigiCollection & trigPrims,
					 EBDigiCollection & barrelDigis,
					 EEDigiCollection & endcapDigis){
  EBDigiCollection selectedBarrelDigis;
  EEDigiCollection selectedEndcapDigis;
  
  run(eventSetup, trigPrims, barrelDigis, endcapDigis,
      &selectedBarrelDigis, &selectedEndcapDigis, nullptr, nullptr);
  
//replaces the input with the suppressed version
  barrelDigis.swap(selectedBarrelDigis);
  endcapDigis.swap(selectedEndcapDigis);  
}


void
EcalSelectiveReadoutSuppressor::run(const edm::EventSetup& eventSetup,
				    const EcalTrigPrimDigiCollection & trigPrims,
				    const EBDigiCollection & barrelDigis,
				    const EEDigiCollection & endcapDigis,
				    EBDigiCollection* selectedBarrelDigis,
				    EEDigiCollection* selectedEndcapDigis,
				    EBSrFlagCollection* ebSrFlags,
				    EESrFlagCollection* eeSrFlags){
  ++ievt_;
  if(!trigPrimBypass_ || ttThresOnCompressedEt_){//uses output of TPG emulator
    setTtFlags(trigPrims);
  } else{//debug mode, run w/o TP digis
    setTtFlags(eventSetup, barrelDigis, endcapDigis);
  }

  ecalSelectiveReadout->runSelectiveReadout0(ttFlags);  

  if(selectedBarrelDigis){
    selectedBarrelDigis->reserve(barrelDigis.size()/20);
    
    // do barrel first
    for(EBDigiCollection::const_iterator digiItr = barrelDigis.begin();
	digiItr != barrelDigis.end(); ++digiItr){
      int interestLevel
	= ecalSelectiveReadout->getCrystalInterest(EBDigiCollection::DetId(digiItr->id())) && ~EcalSelectiveReadout::FORCED_MASK;
      if(accept(*digiItr, zsThreshold[BARREL][interestLevel])){
	selectedBarrelDigis->push_back(digiItr->id(), digiItr->begin());
      } 
    }
  }
  
  // and endcaps
  if(selectedEndcapDigis){
    selectedEndcapDigis->reserve(endcapDigis.size()/20);
    for(EEDigiCollection::const_iterator digiItr = endcapDigis.begin();
	digiItr != endcapDigis.end(); ++digiItr){
      int interestLevel
        = ecalSelectiveReadout->getCrystalInterest(EEDigiCollection::DetId(digiItr->id()))
        & ~EcalSelectiveReadout::FORCED_MASK;
      if(accept(*digiItr, zsThreshold[ENDCAP][interestLevel])){
	selectedEndcapDigis->push_back(digiItr->id(), digiItr->begin());
      }
    }
  }

   if(ievt_ <= 10){
     int neb = (selectedBarrelDigis?selectedBarrelDigis->size():0);
     if(selectedEndcapDigis) LogDebug("EcalSelectiveReadout")
			       //       << __FILE__ << ":" << __LINE__ << ": "
       << "Number of EB digis passing the SR: " << neb
       << " / " << barrelDigis.size() << "\n";
     if(selectedEndcapDigis) LogDebug("EcalSelectiveReadout")
			       //       << __FILE__ << ":" << __LINE__ << ": "
       << "\nNumber of EE digis passing the SR: "
       << selectedEndcapDigis->size()
       << " / " << endcapDigis.size() << "\n";
   }
  
  if(ebSrFlags) ebSrFlags->reserve(34*72);
  if(eeSrFlags) eeSrFlags->reserve(624);
  //SR flags:
  for(int iZ = -1; iZ <=1; iZ+=2){ //-1=>EE-, EB-, +1=>EE+, EB+
    //barrel:
    for(unsigned iEta = 1; iEta <= nBarrelTriggerTowersInEta/2; ++iEta){
      for(unsigned iPhi = 1; iPhi <= nTriggerTowersInPhi; ++iPhi){
	const EcalTrigTowerDetId id(iZ, EcalBarrel, iEta, iPhi);
	EcalSelectiveReadout::towerInterest_t interest
	  = ecalSelectiveReadout->getTowerInterest(id);
	if(interest<0){
	  throw cms::Exception("EcalSelectiveReadout")
	    << __FILE__ << ":" << __LINE__ << ": " << "unknown SR flag. for "
	    << " TT " << id << ". Most probably a bug.";
	}
	int flag;
	//	if(interest==EcalSelectiveReadout::FORCED_RO){
	//  flag = EcalSrFlag::SRF_FORCED_MASK | EcalSrFlag::SRF_FULL;
	//} else{
	flag = srFlags[BARREL][interest];
	//}
	if(ebSrFlags) ebSrFlags->push_back(EBSrFlag(id, flag));
      }//next iPhi
    } //next barrel iEta

    //endcap:
    EcalScDetId id;
    for(int iX = 1; iX <= 20; ++iX){
      for(int iY = 1; iY <= 20; ++iY){

	if (EcalScDetId::validDetId(iX, iY, iZ))
	  id = EcalScDetId(iX, iY, iZ);
	else
	  continue;
	
	EcalSelectiveReadout::towerInterest_t interest
	  = ecalSelectiveReadout->getSuperCrystalInterest(id);
	
	if(interest>=0){//negative no SC at (iX,iY) coordinates
	  int flag;
	  //	  if(interest==EcalSelectiveReadout::FORCED_RO){
	  //  flag = EcalSrFlag::SRF_FORCED_MASK | EcalSrFlag::SRF_FULL;
	  //} else{
	  flag = srFlags[ENDCAP][interest];
	  //}
	  if(eeSrFlags) eeSrFlags->push_back(EESrFlag(id, flag));
	} else  if(iX < 9 || iX > 12 || iY < 9 || iY >12){ //not an inner partial SC
      	  edm::LogError("EcalSelectiveReadout") << __FILE__ << ":" << __LINE__ << ": "
						<<  "negative interest in EE for SC "
						<< id << "\n";
	}
      } //next iY
    } //next iX
  }
}


void EcalSelectiveReadoutSuppressor::setTtFlags(const EcalTrigPrimDigiCollection & trigPrims){
  for(size_t iEta0 = 0; iEta0 < nTriggerTowersInEta; ++iEta0){
    for(size_t iPhi0 = 0; iPhi0 < nTriggerTowersInPhi; ++iPhi0){
      ttFlags[iEta0][iPhi0] = defaultTtf_;
    }
  }
  for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims.begin();
      trigPrim != trigPrims.end(); ++trigPrim){
    int iEta =  trigPrim->id().ieta();
    unsigned int iEta0;
    if(iEta<0){ //z- half ECAL: transforming ranges -28;-1 => 0;27
      iEta0 = iEta + nTriggerTowersInEta/2;
    } else{ //z+ halfECAL: transforming ranges 1;28 => 28;55
      iEta0 = iEta + nTriggerTowersInEta/2 - 1;
    }

    unsigned int iPhi0 = trigPrim->id().iphi() - 1;

    if(!ttThresOnCompressedEt_){
      ttFlags[iEta0][iPhi0] =
        (EcalSelectiveReadout::ttFlag_t) trigPrim->ttFlag();
    } else{
      int compressedEt = trigPrim->compressedEt();
      if(compressedEt < trigPrimBypassLTH_){
        ttFlags[iEta0][iPhi0] = EcalSelectiveReadout::TTF_LOW_INTEREST;
      } else if(compressedEt < trigPrimBypassHTH_){
        ttFlags[iEta0][iPhi0] = EcalSelectiveReadout::TTF_MID_INTEREST;
      } else{
        ttFlags[iEta0][iPhi0] = EcalSelectiveReadout::TTF_HIGH_INTEREST;
      }
    }
  }
}


vector<int> EcalSelectiveReadoutSuppressor::getFIRWeigths() {
  if(firWeights.empty()){
    firWeights = vector<int>(nFIRTaps, 0); //default weight: 0;
    const static int maxWeight = 0xEFF; //weights coded on 11+1 signed bits
    for(unsigned i=0; i < min((size_t)nFIRTaps,weights.size()); ++i){ 
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
//  static const CaloSubdetectorGeometry* eeGeometry = 0;
//  static const CaloSubdetectorGeometry* ebGeometry = 0;
  const CaloSubdetectorGeometry* eeGeometry = nullptr;
  const CaloSubdetectorGeometry* ebGeometry = nullptr;
//  if(eeGeometry==0 || ebGeometry==0){
    edm::ESHandle<CaloGeometry> geoHandle;
    es.get<CaloGeometryRecord>().get(geoHandle);
    eeGeometry
      = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    ebGeometry
      = (*geoHandle).getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
//  }

  //init trigPrim array:
  bzero(trigPrim, sizeof(trigPrim));
	
  for(EBDigiCollection::const_iterator it = ebDigis.begin();
      it != ebDigis.end(); ++it){
    EBDataFrame frame(*it);
    const EcalTrigTowerDetId& ttId = theTriggerMap->towerOf(frame.id());
//      edm:::LogDebug("TT") << __FILE__ << ":" << __LINE__ << ": "
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
    EEDataFrame frame(*it);
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

  double gainInv[] = {12., 1., 6., 12.};


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

void EcalSelectiveReadoutSuppressor::printTTFlags(ostream& os, int iEvent,
						  bool withHeader) const{
  const char tccFlagMarker[] = { '?', '.', 'S', '?', 'C', 'E', 'E', 'E', 'E'};
  const int nEta = EcalSelectiveReadout::nTriggerTowersInEta;
  const int nPhi = EcalSelectiveReadout::nTriggerTowersInPhi;
  
  if(withHeader){
    os << "# TCC flag map\n#\n"
      "# +-->Phi            " << tccFlagMarker[1+0] << ": 000 (low interest)\n"
      "# |                  " << tccFlagMarker[1+1] << ": 001 (mid interest)\n"
      "# |                  " << tccFlagMarker[1+2] << ": 010 (not valid)\n"
      "# V Eta              " << tccFlagMarker[1+3] << ": 011 (high interest)\n"
      "#                    " << tccFlagMarker[1+4] << ": 1xx forced readout (Hw error)\n";
  }

  if(iEvent>=0){
    os << "#\n#Event " << iEvent << "\n";
  }
  
  for(int iEta=0; iEta<nEta; ++iEta){
    for(int iPhi=0; iPhi<nPhi; ++iPhi){
      os << tccFlagMarker[ttFlags[iEta][iPhi]+1];
    }
    os << "\n";
  }
}
