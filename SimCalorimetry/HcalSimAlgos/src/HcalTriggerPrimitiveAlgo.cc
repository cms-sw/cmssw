#include "SimCalorimetry/HcalSimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSimParameterMap.h"
#include <iostream>
using namespace cms;

inline double theta_from_eta(double eta){return (2.0*atan(exp(-eta)));}

HcalTriggerPrimitiveAlgo::HcalTriggerPrimitiveAlgo()
: theThreshold(0.5)
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
  std::cout << "[HcalTriggerPrimitiveAlgo] Calibration constants: " << theHBHECalibrationConstant << " " << theHFCalibrationConstant << std::endl;
}


HcalTriggerPrimitiveAlgo::~HcalTriggerPrimitiveAlgo()
{
}


void HcalTriggerPrimitiveAlgo::run(const HBHEDigiCollection & hbheDigis,
                                const HFDigiCollection & hfDigis,
                                HcalTrigPrimRecHitCollection & result)
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

  for(SumMap::const_iterator mapItr = theSumMap.begin(); mapItr != theSumMap.end(); ++mapItr)
  {
    analyze(mapItr->second, result);
  }
  
  return;
}


void HcalTriggerPrimitiveAlgo::addSignal(const HBHEDataFrame & frame) {
  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
  assert(ids.size() == 1 || ids.size() == 2);
  CaloSamples samples1(ids[0], frame.size());
  theCoder.adc2fC(frame, samples1);
  transverseComponent(samples1, ids[0]);

  if(ids.size() == 2) {
    // make a second trigprim for the other one, and split the energy
    CaloSamples samples2(ids[1], samples1.size());
    for(int i = 0; i < samples1.size(); ++i) {
      samples1[i] *= 0.5;
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
    CaloSamples samples(ids[0], frame.size());
    theCoder.adc2fC(frame, samples);
    transverseComponent(samples, ids[0]);
    addSignal(samples);
  }
}


void HcalTriggerPrimitiveAlgo::transverseComponent(CaloSamples & samples, 
                                                   const HcalTrigTowerDetId & id) const 
{
  // the adc2fC overwrites the Samples' DetId with a DataFrame id.  We want a
  // trig tower ID.  so we need to make a copy and switch
  CaloSamples result(id, samples.size());
  // CaloSampels doesnt have a *= method
  for(int i = 0; i < samples.size(); ++i) {
    result[i] = samples[i] * theSinThetaTable[abs(id.ieta())];
  }
  // swap it in
  samples = result;
}


void HcalTriggerPrimitiveAlgo::addSignal(const CaloSamples & samples) {
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


void HcalTriggerPrimitiveAlgo::analyze(const CaloSamples & samples, 
                                    HcalTrigPrimRecHitCollection & result) const 
{
  // look for local maxima over threshold
  for(int ibin = 1; ibin < samples.size(); ++ibin) {
    // number of trigprims from this sample
    int n = 0; 
    if(samples[ibin] > samples[ibin-1] && samples[ibin] > samples[ibin+1]) {
      // now compare ET to threshold.  ET will be the sum of two bins in HB/HR/HO, and one bin in HF
      HcalTrigTowerDetId detId(samples.id());
      double et;
      if(detId.ietaAbs() > theTrigTowerGeometry.firstHFTower()) {
        et = samples[ibin] * theHFCalibrationConstant;
      } else {
        // correct for charge out of the time window
        et = (samples[ibin] + samples[ibin+1]) * 1.139 * theHBHECalibrationConstant;
      }
   
      if(et > theThreshold) {
        // signal should be in 5th time bin
        int bunch = ibin-5;
        ++n;
        HcalTriggerPrimitiveRecHit trigPrim(detId, et, 0., bunch, 0, n);
std::cout << trigPrim << std::endl;
        result.push_back( trigPrim );
      }
    }
  }
}


