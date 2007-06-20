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

EcalSelectiveReadoutSuppressor::EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params):
  firstFIRSample(params.getParameter<int>("ecalDccZs1stSample")),
  weights(params.getParameter<vector<double> >("dccNormalizedWeights")){
  
  double adcToGeV = params.getParameter<double>("ebDccAdcToGeV");
  thrUnit[BARREL] = adcToGeV/4.; //unit=1/4th ADC count
  
  adcToGeV = params.getParameter<double>("eeDccAdcToGeV");
  thrUnit[ENDCAP] = adcToGeV/4.; //unit=1/4th ADC count
  ecalSelectiveReadout
    = auto_ptr<EcalSelectiveReadout>(new EcalSelectiveReadout
				     (params.getParameter<int>("deltaEta"),
				      params.getParameter<int>("deltaPhi")));
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


void EcalSelectiveReadoutSuppressor::setTriggerMap(const EcalTrigTowerConstituentsMap * map){
  theTriggerMap = map;
  ecalSelectiveReadout->setTriggerMap(map);
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
  int lowInterestSrFlag[2];
  int highInterestThr[2];
  int highInterestSrFlag[2];
  
  lowInterestThr[BARREL] = internalThreshold(barrelLowInterest, BARREL);
  lowInterestSrFlag[BARREL] = thr2Srf(lowInterestThr[BARREL],
				      EcalSrFlag::SRF_ZS1);
  
  highInterestThr[BARREL] = internalThreshold(barrelHighInterest, BARREL);
  highInterestSrFlag[BARREL] = thr2Srf(highInterestThr[BARREL],
				       EcalSrFlag::SRF_ZS2);
  
  lowInterestThr[ENDCAP] = internalThreshold(endcapLowInterest, ENDCAP);
  lowInterestSrFlag[ENDCAP] = thr2Srf(lowInterestThr[ENDCAP],
				      EcalSrFlag::SRF_ZS2);

  highInterestThr[ENDCAP] = internalThreshold(endcapHighInterest, ENDCAP); 
  highInterestSrFlag[ENDCAP] = thr2Srf(highInterestThr[ENDCAP],
				       EcalSrFlag::SRF_ZS2);
  
  for(int iSubDet = 0; iSubDet<2; ++iSubDet){
    //low interest
    zsThreshold[iSubDet][0] = lowInterestThr[iSubDet];
    srFlags[iSubDet][0] = lowInterestSrFlag[iSubDet];

    //single->high interest
    zsThreshold[iSubDet][1] = highInterestThr[iSubDet];
    srFlags[iSubDet][1] = highInterestSrFlag[iSubDet];

    //neighbour->high interest
    zsThreshold[iSubDet][2] = highInterestThr[iSubDet];
    srFlags[iSubDet][2] = highInterestSrFlag[iSubDet];

    //center->high interest
    zsThreshold[iSubDet][3] = highInterestThr[iSubDet];
    srFlags[iSubDet][3] = highInterestSrFlag[iSubDet];
  }
}

int EcalSelectiveReadoutSuppressor::thr2Srf(int thr, int zsFlag) const{
  if(thr==numeric_limits<int>::max()){
    return EcalSrFlag::SRF_SUPPRESS;
  }
  if(thr==numeric_limits<int>::min()){
    return EcalSrFlag::SRF_FULL;
  } 
  return zsFlag;
}

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
template<class T>
bool EcalSelectiveReadoutSuppressor::accept(const T& frame,
					    int thr){
  //FIR filter weights:
  const vector<int>& w = getFIRWeigths();
  
  //accumulator used to compute weighted sum of samples
  int acc = 0;
  bool gain12saturated = false;
  const int gain12 = 0x01; 
  const int lastFIRSample = firstFIRSample + nFIRTaps - 1;
  //LogDebug("DccFir") << "DCC FIR operation: ";
  for(int i=firstFIRSample-1; i<lastFIRSample; ++i){
    if(i>=0 && i < frame.size()){
      const EcalMGPASample& sample = frame[i];
      if(sample.gainId()!=gain12) gain12saturated = true;
      //LogTrace("DccFir") << (i>=firstFIRSample?"+":"") << sample.adc()
      // 			  << "*(" << w[i] << ")";
      acc+=sample.adc()*w[i];
    } else{
      edm::LogWarning("DccFir") << __FILE__ << ":" << __LINE__ <<
	": Not enough samples in data frame or 'ecalDccZs1stSample' module "
	"parameter is not valid...";
    }
  }
  //LogTrace("DccFir") << "\n";
  //discards the 8 LSBs 
  //(shift operator cannot be used on negative numbers because
  // the result depends on compilator implementation)
  acc = (acc>=0)?(acc >> 8):-(-acc >> 8);
  //ZS passed if weigthed sum acc above ZS threshold or if
  //one sample has a lower gain than gain 12 (that is gain 12 output
  //is saturated)

  const bool result = acc>=thr || gain12saturated;
  
  //LogTrace("DccFir") << "acc: " << acc << "\n"
  //  		     << "threshold: " << thr << " ("
  //  		     << thr*thrUnit[frame.id().subdet()==EcalBarrel?0:1]
  //  		     << "GeV)\n"
  //  		     << "saturated: " << (gain12saturated?"yes":"no") << "\n"
  //  		     << "ZS passed: " << (result?"yes":"no") << "\n";

  return result;
}

void EcalSelectiveReadoutSuppressor::run(const edm::EventSetup& eventSetup,   
					 const EcalTrigPrimDigiCollection & trigPrims,
					 EBDigiCollection & barrelDigis,
					 EEDigiCollection & endcapDigis){
  EBDigiCollection selectedBarrelDigis;
  EEDigiCollection selectedEndcapDigis;
  EBSrFlagCollection ebSrFlags;
  EESrFlagCollection eeSrFlags;
  
  run(eventSetup, trigPrims, barrelDigis, endcapDigis,
      selectedBarrelDigis, selectedEndcapDigis, ebSrFlags, eeSrFlags);
  
//replaces the input with the suppressed version
  barrelDigis.swap(selectedBarrelDigis);
  endcapDigis.swap(selectedEndcapDigis);  
}


void
EcalSelectiveReadoutSuppressor::run(const edm::EventSetup& eventSetup,
				    const EcalTrigPrimDigiCollection & trigPrims,
				    const EBDigiCollection & barrelDigis,
				    const EEDigiCollection & endcapDigis,
				    EBDigiCollection& selectedBarrelDigis,
				    EEDigiCollection& selectedEndcapDigis,
				    EBSrFlagCollection& ebSrFlags,
				    EESrFlagCollection& eeSrFlags){
  if(!trigPrimBypass_){//normal mode
    setTtFlags(trigPrims);
  } else{//debug mode, run w/o TP digis
    setTtFlags(eventSetup, barrelDigis, endcapDigis);
  }

  ecalSelectiveReadout->runSelectiveReadout0(ttFlags);  
  
  selectedBarrelDigis.reserve(barrelDigis.size()/20);
  selectedEndcapDigis.reserve(endcapDigis.size()/20);

  // do barrel first
  for(EBDigiCollection::const_iterator digiItr = barrelDigis.begin();
      digiItr != barrelDigis.end(); ++digiItr){
    int interestLevel
      = ecalSelectiveReadout->getCrystalInterest(digiItr->id());
    if(accept(*digiItr, zsThreshold[BARREL][interestLevel])){
      selectedBarrelDigis.push_back(*digiItr);
    } 
  }
  
  // and endcaps
  for(EEDigiCollection::const_iterator digiItr = endcapDigis.begin();
      digiItr != endcapDigis.end(); ++digiItr){
    int interestLevel = ecalSelectiveReadout->getCrystalInterest(digiItr->id());
    if(accept(*digiItr, zsThreshold[ENDCAP][interestLevel])){
      selectedEndcapDigis.push_back(*digiItr);
    }
  }
  
  ebSrFlags.reserve(34*72);
  eeSrFlags.reserve(624);
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
	if(interest==EcalSelectiveReadout::FORCED_RO){
	  flag = EcalSrFlag::SRF_FORCED_MASK | EcalSrFlag::SRF_FULL;
	} else{
	  flag = srFlags[BARREL][interest];
	}
	ebSrFlags.push_back(EBSrFlag(id, flag));
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
	  if(interest==EcalSelectiveReadout::FORCED_RO){
	    flag = EcalSrFlag::SRF_FORCED_MASK | EcalSrFlag::SRF_FULL;
	  } else{
	    flag = srFlags[BARREL][interest];
	  }
	  eeSrFlags.push_back(EESrFlag(id, flag));
	} else{
	  cout << __FILE__ << ":" << __LINE__ << ": "
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
      ttFlags[iEta0][iPhi0] = EcalSelectiveReadout::TTF_FORCED_RO_OTHER1;
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
    ttFlags[iEta0][iPhi0] =
      (EcalSelectiveReadout::ttFlag_t) trigPrim->ttFlag();
  }
}


vector<int> EcalSelectiveReadoutSuppressor::getFIRWeigths() {
  if(firWeights.size()==0){
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
  
