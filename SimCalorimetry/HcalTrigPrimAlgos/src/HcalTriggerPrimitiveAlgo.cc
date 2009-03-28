#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
using namespace std;

HcalTriggerPrimitiveAlgo::HcalTriggerPrimitiveAlgo(bool pf, const std::vector<double>& w,
						   int latency, uint32_t FG_threshold, uint32_t ZS_threshold, int firstTPSample, int TPSize)
  : incoder_(0), outcoder_(0), theThreshold(0),
    peakfind_(pf), weights_(w), latency_(latency), FG_threshold_(FG_threshold), ZS_threshold_(ZS_threshold), firstTPSample_(firstTPSample), TPSize_(TPSize)
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
  theFGSumMap.clear();
  theTowerMapFG.clear();
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
   //Hack for 300_pre10, should be removed.
   if (frame.id().depth()==5) return;

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

 
  if(frame.id().depth() == 1 || frame.id().depth() == 2) {
    std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
    assert(ids.size() == 1);
    IntegerCaloSamples samples(ids[0], frame.size());
    samples.setPresamples(frame.presamples());
    // for(int i = 0; i < frame.size(); i++)
    // {cout<<frame.sample(i).adc()<<" ";}
    //cout<<endl;
    incoder_->adc2Linear(frame, samples);
    
    addSignal(samples);
        
    uint32_t fgid;

    // Mask off depths: fgid is the same for both depths

    fgid = (frame.id().rawId() | 0x1c000) ;
	 
    SumMapFG::iterator itr = theFGSumMap.find(fgid);
   
    if(itr == theFGSumMap.end()) {
      theFGSumMap.insert(std::make_pair(fgid, samples));
    } 
    else {
      // wish CaloSamples had a +=
      for(int i = 0; i < samples.size(); ++i) {
	(itr->second)[i] += samples[i];        
      }      
    }

    // Depth =2 is the second entry in map (sum). Use its original Hcal Det Id to obtain trigger tower
    if (frame.id().depth()==2)
      {      
      for(unsigned int n = 0; n < ids.size(); n++)
      	  {
          theTowerMapFG.insert(TowerMapFG::value_type(ids[n],itr->second));
	  }
      }
    
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
  //std::vector<bool> finegrain(outlength,false);
  std::vector<bool> finegrain(TPSize_,false);
  IntegerCaloSamples sum(samples.id(), outlength);
  
  bool SOI_pegged =  false;
  //Test is SOI input is pegged before summing
  if(samples[samples.presamples()]> 0x3FF) SOI_pegged = true;
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
      //IntegerCaloSamples output(samples.id(),outlength-2);
      IntegerCaloSamples output(samples.id(),TPSize_);
      //output.setPresamples(newprelength-1);
      output.setPresamples(newprelength - firstTPSample_);
      //for(int ibin2 = 1; ibin2 < (sum.size())-1; ++ibin2) 
      for(int ibin2 = 0; ibin2 < TPSize_; ++ibin2) 
	{
	  //use if peak finding true
        int idx = firstTPSample_ + ibin2;
        //if ( samples[ibin2] >= samples[ibin2-1] && samples[ibin2] > samples[ibin2+1] && samples[ibin2] > theThreshold)
        if ( samples[idx] >= samples[idx-1] && samples[idx] > samples[idx+1] && samples[idx] > theThreshold)
	    {
	      //output[ibin2-1]=sum[ibin2];//if peak found
	      output[ibin2]=sum[idx];//if peak found
	    }
	  //else{output[ibin2-1]=0;}//if no peak
	  else{output[ibin2]=0;}//if no peak
	}
      if(SOI_pegged == true)
	{
	  output[output.presamples()] = 0x3FF;
	}
      outcoder_->compress(output, finegrain, result);//send to transcoder
    }
  
  else//No peak finding
    {
      //IntegerCaloSamples output(samples.id(),outlength);
      //output.setPresamples(newprelength);
      IntegerCaloSamples output(samples.id(),TPSize_);
      output.setPresamples(newprelength - firstTPSample_ +1);
      //for(int ibin2 = 0; ibin2<sum.size(); ++ibin2) 
      for(int ibin2 = 0; ibin2 < TPSize_; ++ibin2) 
	{
	  output[ibin2]=sum[ibin2+firstTPSample_];//just pass value
	}
      outcoder_->compress(output, finegrain, result);
      runZS(result);
    }   
}


void HcalTriggerPrimitiveAlgo::analyzeHF(IntegerCaloSamples & samples, 
					 HcalTriggerPrimitiveDigi & result)
{
  //std::vector<bool> finegrain(samples.size(),false);
  std::vector<bool> finegrain(TPSize_,false);
  // IntegerCaloSamples sum(samples.id(), samples.size());
  HcalTrigTowerDetId detId_(samples.id()); 
   
  // get information from Tower map
  for(TowerMapFG::iterator mapItr = theTowerMapFG.begin(); mapItr != theTowerMapFG.end(); ++mapItr)
    {

      HcalTrigTowerDetId detId(mapItr->first);
      if (detId == detId_) {
	//for (int i=0; i < samples.size(); ++i) {
	for (int i=firstTPSample_; i < firstTPSample_+TPSize_; ++i) {
	  bool set_fg = false;
	  mapItr->second[i] >= FG_threshold_ ? set_fg = true : false;
	  finegrain[i - firstTPSample_] = (finegrain[i - firstTPSample_] || set_fg);
	}
      }
    }  
  //IntegerCaloSamples output(samples.id(),samples.size());
  IntegerCaloSamples output(samples.id(),TPSize_);
  //output.setPresamples(samples.presamples());
  output.setPresamples(samples.presamples() - firstTPSample_);
  //cout<<"Presamples = "<<samples.presamples()<<endl;
 // for(int ibin2 = 0; ibin2 < samples.size(); ++ibin2) 
     for(int ibin2 = 0; ibin2 < TPSize_; ++ibin2) 
    {//output[ibin2]=sum[ibin2];
      //samples[ibin2] /= 4;  // for 0.25 GeV ET RCT LSB
      //cout << "samples: " << samples[i] << endl;
      //if (samples[ibin2] > 0x3FF) samples[ibin2] = 0x3FF;  //Compression is 1 to 1 with saturation at 8 bits
      output[ibin2]=samples[ibin2+firstTPSample_]/4;
      if (output[ibin2] > 0x3FF) output[ibin2] = 0x3FF;  //Compression is 1 to 1 with saturation at 8 bits

    }
  //cout<<endl;
  outcoder_->compress(output, finegrain, result);
  runZS(result);
}

void HcalTriggerPrimitiveAlgo::runZS(HcalTriggerPrimitiveDigi & tp){
   bool ZS = true;
   for (int i=0; i<tp.size(); ++i){
      if (tp[i].compressedEt()  > ZS_threshold_) {
         ZS=false;
         break;
      }
   }
   if (ZS) tp.setZSInfo(false,true);
   else tp.setZSInfo(true,false);
}
