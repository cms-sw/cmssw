#include "SimCalorimetry/HcalSimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

HcalTriggerPrimitiveAlgo::HcalTriggerPrimitiveAlgo(const HcalCoderFactory * coderFactory)
  : incoder_(0), outcoder_(0), theCoderFactory(coderFactory), theThreshold(0.5)
{
}


HcalTriggerPrimitiveAlgo::~HcalTriggerPrimitiveAlgo()
{
}


void HcalTriggerPrimitiveAlgo::run(const HBHEDigiCollection & hbheDigis,
				   const HFDigiCollection & hfDigis,
				   HcalTrigPrimDigiCollection & result)
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
      result.push_back(HcalTriggerPrimitiveDigi(mapItr->first));
      analyze(mapItr->second, result.back());
    }

  theSumMap.clear();  
  return;
}


void HcalTriggerPrimitiveAlgo::addSignal(const HBHEDataFrame & frame) {

  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
  assert(ids.size() == 1 || ids.size() == 2);
  IntegerCaloSamples samples1(ids[0], int(frame.size()));
  
  incoder_->adc2ET(frame, samples1);
  
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
    
    incoder_->adc2ET(frame, samples);
    
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

void HcalTriggerPrimitiveAlgo::analyze(IntegerCaloSamples & samples, 
				       HcalTriggerPrimitiveDigi & result)
{
  std::vector<bool> finegrain;
  std::vector <uint32_t> sampEt;

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
  result.setSize(samples.size());
  for(int ibin = 1; ibin < samples.size()-1; ++ibin)
    {
      outcoder_->htrCompress(samples, finegrain, result); 
    }

}











