#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
using namespace std;

HcalTriggerPrimitiveAlgo::HcalTriggerPrimitiveAlgo(bool pf, const std::vector<double>& w,
						   int latency)
  : incoder_(0), outcoder_(0), theThreshold(0),
    peakfind_(pf), weights_(w), latency_(latency)
{
}


HcalTriggerPrimitiveAlgo::~HcalTriggerPrimitiveAlgo()
{
}


void HcalTriggerPrimitiveAlgo::run(const HcalTPGCoder * incoder,
				   const HcalTPGCompressor * outcoder,
				   const HBHEDigiCollection & hbheDigis,
				   const HFDigiCollection & hfDigis,
				   HcalTrigPrimDigiCollection & result)
{

  incoder_=incoder;
  outcoder_=outcoder;

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
      HcalTrigTowerDetId detId(mapItr->second.id());
      if(detId.ietaAbs() >= theTrigTowerGeometry.firstHFTower())
	{ analyzeHF(mapItr->second, result.back());}
      else{analyze(mapItr->second, result.back());}
    }

  theSumMap.clear();  
  return;
}


void HcalTriggerPrimitiveAlgo::addSignal(const HBHEDataFrame & frame) {

  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
  assert(ids.size() == 1 || ids.size() == 2);
  IntegerCaloSamples samples1(ids[0], int(frame.size()));

  samples1.setPresamples(frame.presamples());
  incoder_->adc2Linear(frame, samples1);
  
  if(ids.size() == 2) {
    // make a second trigprim for the other one, and split the energy
    IntegerCaloSamples samples2(ids[1], samples1.size());
    for(int i = 0; i < samples1.size(); ++i) 
      {
	samples1[i] = uint32_t(samples1[i]*0.5);
	samples2[i] = samples1[i];
      }
    samples2.setPresamples(frame.presamples());
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
    samples.setPresamples(frame.presamples());
    // for(int i = 0; i < frame.size(); i++)
    // {cout<<frame.sample(i).adc()<<" ";}
    //cout<<endl;
    incoder_->adc2Linear(frame, samples);
    
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

  //cout<<"TPG calc"
  // HcalTrigTowerDetId detId(samples.id());
  //find output samples length and new presamples
  int shrink = weights_.size()-1; //REAL LINE
  int outlength=samples.size() - shrink;
  int newprelength = ((samples.presamples()+1)-weights_.size())+latency_;
  std::vector<bool> finegrain(outlength,false);
  IntegerCaloSamples sum(samples.id(), outlength);

  
  bool highvalue = false;
  //slide algo window
  for(int ibin = 0; ibin < int(samples.size())- shrink; ++ibin)
    {
      if(samples[ibin+1]>10){highvalue = true;}
      int algosumvalue = 0;
      for(unsigned int i = 0; i < weights_.size(); i++) {
	algosumvalue += int(samples[ibin+i] * weights_[i]);
      }//add up value * scale factor
      if (algosumvalue<0) sum[ibin]=0; // low-side
      else if (algosumvalue>0x3FF) sum[ibin]=0x3FF;  //high-side
      else sum[ibin] = algosumvalue;//assign value to sum[]
    }

  if(highvalue)
    {
      // cout<<"ICS ";
      for(int ibin = 0; ibin < int(samples.size())- shrink; ++ibin)
	{
	  //  cout<<samples[ibin+1]<<" ";
	}
      // cout<<endl;
    }
  //Do peak finding if requested


  if(peakfind_)
    {
      IntegerCaloSamples output(samples.id(),outlength-2);
      output.setPresamples(newprelength-1);
      for(int ibin2 = 1; ibin2 < (sum.size())-2; ++ibin2) 
	{
	  //use if peak finding true
	  //Old equalities
	  // if ( sum[ibin2] > sum[ibin2-1] && 
	  //    sum[ibin2] >= sum[ibin2+1] && 
	  //    sum[ibin2] > theThreshold)
	  if ( sum[ibin2] >= sum[ibin2-1] && 
	              sum[ibin2] > sum[ibin2+1] && 
	       sum[ibin2] > theThreshold)
	    {
	      output[ibin2-1]=sum[ibin2];//if peak found
	    }
	  else{output[ibin2-1]=0;}//if no peak
	}
      outcoder_->compress(output, finegrain, result);//send to transcoder
      //      outcoder_->loadhcalUncompress();
    }
  
  else//No peak finding
    {
      IntegerCaloSamples output(samples.id(),outlength);
      output.setPresamples(newprelength);
      for(int ibin2 = 0; ibin2<sum.size(); ++ibin2) 
	{
	  output[ibin2]=sum[ibin2];//just pass value
	}
      outcoder_->compress(output, finegrain, result);
    }   
}


void HcalTriggerPrimitiveAlgo::analyzeHF(IntegerCaloSamples & samples, 
					 HcalTriggerPrimitiveDigi & result)
{
  std::vector<bool> finegrain(samples.size(),false);
  IntegerCaloSamples sum(samples.id(), samples.size());
  
  
  IntegerCaloSamples output(samples.id(),samples.size());
  output.setPresamples(samples.presamples());
  //cout<<"Presamples = "<<samples.presamples()<<endl;
  for(int ibin2 = 0; ibin2 < samples.size(); ++ibin2) 
    {//output[ibin2]=sum[ibin2];
      output[ibin2]=samples[ibin2];
    }
  //cout<<endl;
  outcoder_->compress(output, finegrain, result);
}
