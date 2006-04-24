



#include "SimCalorimetry/HcalSimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <iostream>

inline double theta_from_eta(double eta){return (2.0*atan(exp(-eta)));}

HcalTriggerPrimitiveAlgo::HcalTriggerPrimitiveAlgo(const HcalCoderFactory * coderFactory)
  : theCoderFactory(coderFactory),
    theThreshold(0.5)
{
  // set up the table of sin(theta) for fast ET calculations
  int nTowers = theTrigTowerGeometry.nTowers();
  for(int itower = 1; itower <= nTowers; ++itower) {
    double eta1, eta2;
    theTrigTowerGeometry.towerEtaBounds(itower, eta1,eta2);
    double eta = 0.5 * (eta1+eta2);
    theSinThetaTable[itower] = sin(theta_from_eta(eta));
  }

  // fill in some nominal calibration constants
  HcalSimParameterMap parameterMap;
  theHBHECalibrationConstant = parameterMap.hbheParameters().calibrationConstant();
  theHFCalibrationConstant = parameterMap.hfParameters1().calibrationConstant();
  LogDebug("HcalTriggerPrimitiveAlgo") << " Calibration constants: " 
				       << theHBHECalibrationConstant << " " << theHFCalibrationConstant;
}


HcalTriggerPrimitiveAlgo::~HcalTriggerPrimitiveAlgo()
{
}


void HcalTriggerPrimitiveAlgo::run(const HBHEDigiCollection & hbheDigis,
				   const HFDigiCollection & hfDigis,
				   HcalTriggerPrimitiveDigi & result)
{

  theSumMap.clear();

  // do the HB/HE digis
  for(HBHEDigiCollection::const_iterator hbheItr = hbheDigis.begin();
      hbheItr != hbheDigis.end(); ++hbheItr)
    {
      addSignal(*hbheItr);
    }

  // and the HF digis
  for(HFDigiCollection::const_iterator hfItr = hfDigis.begin();
      hfItr != hfDigis.end(); ++hfItr)
    {
      addSignal(*hfItr);
    }


  for(SumMap::iterator mapItr = theSumMap.begin(); mapItr != theSumMap.end(); ++mapItr)
    {
      analyze(mapItr->second, result);
      
      
    }
  
  return;
}


void HcalTriggerPrimitiveAlgo::addSignal(const HBHEDataFrame & frame) {

  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
  assert(ids.size() == 1 || ids.size() == 2);
  IntegerCaloSamples samples1(ids[0], int(frame.size()));
  
  //replaced by the transcoder
  //theCoderFactory->coder(frame.id())->adc2fC(frame, samples1);
  //transverseComponent(samples1, ids[0]);
  //HcalTPGCoder::adc2ET(frame, samples1);
  //  HcalTPGCoder tcoder;
  tcoder->adc2ET(frame, samples1);
  
  if(ids.size() == 2) {
    // make a second trigprim for the other one, and split the energy
    IntegerCaloSamples samples2(ids[1], samples1.size());
    for(int i = 0; i < samples1.size(); ++i) 
      {
	samples1[i] = uint32_t(samples1[i]*0.5);
	samples2[i] = samples1[i];
      }
    addSignal(samples2);
  }
  addSignal(samples1);
}


void HcalTriggerPrimitiveAlgo::addSignal(const HFDataFrame & frame) {

  // HF short fibers off
  if(frame.id().depth() == 1) {
    std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
    assert(ids.size() == 1);
    IntegerCaloSamples samples(ids[0], frame.size());
    
    //Replaced by transcoder
    //theCoderFactory->coder(frame.id())->adc2fC(frame, samples);
    //transverseComponent(samples, ids[0]);
    tcoder->adc2ET(frame, samples);
    
    addSignal(samples);
  }
}


void HcalTriggerPrimitiveAlgo::addSignal(const IntegerCaloSamples & samples) {
  HcalTrigTowerDetId id(samples.id());
  SumMap::iterator itr = theSumMap.find(id);
  if(itr == theSumMap.end()) {
    theSumMap.insert(std::make_pair(id, samples));
  } else {
    // wish CaloSamples had a +=
    for(int i = 0; i < samples.size(); ++i) {
      
      (itr->second)[i] += samples[i];
    }
  }
}

/* replaced by transcoder
   void HcalTriggerPrimitiveAlgo::transverseComponent(IntegerCaloSamples & samples, 
   const HcalTrigTowerDetId & id) const 
   {
   // the adc2fC overwrites the Samples' DetId with a DataFrame id.  We want a
   // trig tower ID.  so we need to make a copy and switch
   IntegerCaloSamples result(id, samples.size());
   // CaloSampels doesnt have a *= method
   for(int i = 0; i < samples.size(); ++i) {
   result[i] = samples[i] * theSinThetaTable[abs(id.ieta())];
   }
   // swap it in
   samples = result;
   }
   
*/


void HcalTriggerPrimitiveAlgo::analyze(IntegerCaloSamples & samples, 
				       HcalTriggerPrimitiveDigi & result)
{
  std::vector<bool> finegrain;
  std::vector <uint32_t> sampEt;
  // std::vector <bool> decision;
  HcalTrigTowerDetId detId(samples.id());
  
  for(int ibin = 1; ibin < samples.size()-1; ++ibin)
    {
      if(detId.ietaAbs() > theTrigTowerGeometry.firstHFTower())
	{sampEt[ibin] = samples[ibin];}
      else
	{sampEt[ibin] = samples[ibin]+samples[ibin+1];}
      // sampEt[ibin] = samples[ibin]+samples[ibin+1];
    }
  
  for(int ibin2 = 2; ibin2 < (samples.size())-2; ++ibin2) 
    {
      if(sampEt[ibin2] > sampEt[ibin2-1] && sampEt[ibin2] > sampEt[ibin2+1] && sampEt[ibin2] > theThreshold) 
	//	++n;
	{
	  samples[ibin2] = uint32_t(0);
	  //decision[ibin2] = true;
	}
      //else{decision = false;}
      
    }
  
  outputMaker(samples, result, finegrain);
  
}


void HcalTriggerPrimitiveAlgo::outputMaker(const IntegerCaloSamples & samples, 
					   HcalTriggerPrimitiveDigi & result, 
					   const std::vector<bool> & finegrain)
{
  for(int ibin = 1; ibin < samples.size()-1; ++ibin)
    {
      transcoder->htrCompress(samples, finegrain, result); 
    }

}





















