#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

using namespace std;

HcalTriggerPrimitiveAlgo::HcalTriggerPrimitiveAlgo( bool pf, const std::vector<double>& w, int latency,
                                                    uint32_t FG_threshold1, uint32_t FG_threshold2, uint32_t ZS_threshold,
                                                    int numberOfSamples, int numberOfPresamples,
                                                    int numberOfSamplesHF, int numberOfPresamplesHF,
                                                    uint32_t minSignalThreshold, uint32_t PMT_NoiseThreshold)
                                                   : incoder_(0), outcoder_(0),
                                                   theThreshold(0), peakfind_(pf), weights_(w), latency_(latency),
                                                   FG_threshold1_(FG_threshold1), FG_threshold2_(FG_threshold2), ZS_threshold_(ZS_threshold),
                                                   numberOfSamples_(numberOfSamples),
                                                   numberOfPresamples_(numberOfPresamples),
                                                   numberOfSamplesHF_(numberOfSamplesHF),
                                                   numberOfPresamplesHF_(numberOfPresamplesHF),
                                                   minSignalThreshold_(minSignalThreshold),
                                                   PMT_NoiseThreshold_(PMT_NoiseThreshold),
                                                   peak_finder_algorithm_(2)
{
   //No peak finding setting (for Fastsim)
   if (!peakfind_){
      numberOfSamples_ = 1; 
      numberOfPresamples_ = 0;
      numberOfSamplesHF_ = 1; 
      numberOfPresamplesHF_ = 0;
   }
   // Switch to integer for comparisons - remove compiler warning
   ZS_threshold_I_ = ZS_threshold_;
}


HcalTriggerPrimitiveAlgo::~HcalTriggerPrimitiveAlgo() {
}

int converttoGCTeta(int& eta)
  {
	int GCTeta = 0;

          if (eta == 29 || eta == 30 || eta == 31)
          {
                  GCTeta = 18;
          }
          else if (eta == 32 || eta == 33 || eta == 34)
          {
                  GCTeta = 19;
          }
          else if (eta == 35 || eta == 36 || eta == 37)
          {
                  GCTeta = 20;
          }
          else if (eta == 38 || eta == 39 || eta == 40 || eta == 41)
          {
                 GCTeta = 21;
          }
          else if (eta == -29 || eta == -30 || eta == -31)
           {
                   GCTeta = 3;
           }
           else if (eta == -32 || eta == -33 || eta == -34)
           {
                   GCTeta = 2;
           }
           else if (eta == -35 || eta == -36 || eta == -37)
           {
                   GCTeta = 1;
           }
           else if (eta == -38 || eta == -39 || eta == -40 || eta == -41)
           {
                   GCTeta = 0;
           }
	return GCTeta;
 }

int converttoGCTphi(int& phi)
 {
  
	int GCTphi = 0;

        if (phi == 1 || phi == 71)
          {       
                  GCTphi = 0;
          }
          else if (phi == 3 || phi == 5)
            {     
                  GCTphi = 1;
            }
          else if (phi == 7 || phi == 9)
            {     
                  GCTphi = 2;
            }
          else if (phi == 11 || phi == 13)
            {     
                  GCTphi = 3;
            }
          else if (phi == 15 || phi == 17)
            {     
                  GCTphi = 4;
            }
          else if (phi == 19 || phi == 21)
            {     
                  GCTphi = 5;
            }
          else if (phi == 23 || phi == 25)
            {     
                  GCTphi = 6;
            }
          else if (phi == 27 || phi == 29)
            {     
                  GCTphi = 7;
            }
          else if (phi == 31 || phi == 33)
            {
                  GCTphi = 8;
            }
          else if (phi == 35 || phi == 37)
            {
                  GCTphi = 9;
            }
          else if (phi == 39 || phi == 41)
            {
                  GCTphi = 10;
            }
          else if (phi == 43 || phi == 45)
            {
                  GCTphi = 11;
            }
          else if (phi == 47 || phi == 49)
            {
                  GCTphi = 12;
            }
          else if (phi == 51 || phi == 53)
             {
                  GCTphi = 13;
             }
          else if (phi == 55 || phi == 57)
             {
                  GCTphi = 14;
	     }
          else if (phi == 59 || phi == 61)
             {
                  GCTphi = 15;
             }
          else if (phi == 63 || phi == 65)
             {
                  GCTphi = 16;
             }
          else if (phi == 67 || phi == 69)
             {
                  GCTphi = 17;
             }
	return GCTphi;
 }

int converttoTPGeta(int& GCTeta)
{
	int TPGeta = 0;

	if (GCTeta == 0)
	{
		TPGeta = -32;
	}
	if (GCTeta == 1)
         {
                 TPGeta = -31;
         }
     if (GCTeta == 2)
         {
                 TPGeta = -30;
         }
     if (GCTeta == 3)
         {
                 TPGeta = -29;
         }
     if (GCTeta == 18)
         {
                 TPGeta = 29;
         }
     if (GCTeta == 19)
         {
                 TPGeta = 30;
         }
     if (GCTeta == 20)
         {
                 TPGeta = 31;
         }
     if (GCTeta == 21)
         {
                 TPGeta = 32;
         }

	return TPGeta;
}

int converttoTPGphi(int& GCTphi)
{
	int TPGphi = 0;

	if (GCTphi == 0)
	{
		TPGphi = 1;
	}
	if (GCTphi == 1)
         {
                 TPGphi = 5;
         }
     if (GCTphi == 2)
         {
                 TPGphi = 9;
         }
     if (GCTphi == 3)
         {
                 TPGphi = 13;
         }
     if (GCTphi == 4)
         {
                 TPGphi = 17;
         }
     if (GCTphi == 5)
         {
                 TPGphi = 21;
         }
     if (GCTphi == 6)
         {
                 TPGphi = 25;
         }
     if (GCTphi == 7)
         {
                 TPGphi = 29;
         }
     if (GCTphi == 8)
         {
                 TPGphi = 33;
         }
     if (GCTphi == 9)
         {
                 TPGphi = 37;
         }
     if (GCTphi == 10)
         {
                 TPGphi = 41;
         }
     if (GCTphi == 11)
         {
                 TPGphi = 45;
         }
     if (GCTphi == 12)
         {
                 TPGphi = 49;
         }
     if (GCTphi == 13)
         {
                 TPGphi = 53;
         }
     if (GCTphi == 14)
         {
                 TPGphi = 57;
         }
     if (GCTphi == 15)
         {
                 TPGphi = 61;
         }
     if (GCTphi == 16)
         {
                 TPGphi = 65;
         }
     if (GCTphi == 17)
         {
                 TPGphi = 69;
         }
return TPGphi;
}

int converttoGCTetaFromTPG(int& etaTPG)
{
	int neweta = 0;
         
         if (etaTPG == -32)
         {       
                 neweta = 0;
         }
         if (etaTPG == -31)
          {       
                  neweta = 1;
          } 
         if (etaTPG == -30)
          {       
                  neweta = 2;
          } 
         if (etaTPG == -29)
          {       
                  neweta = 3;
          }
         
         if (etaTPG == 29)
          {       
                  neweta = 18;
          }
          if (etaTPG == 30)
           {       
                   neweta = 19;
           }  
          if (etaTPG == 31)
           {       
                   neweta = 20;
           }  
          if (etaTPG == 32)
           {       
                   neweta = 21;
           }
	return neweta;	
} 

int converttoGCTphiFromTPG(int& phiTPG)
{
	int newphi = 0;
 
          if (phiTPG == 1)
           {
                   newphi = 0;
           }
           else if (phiTPG == 5 )
             {
                   newphi = 1;
             }
           else if (phiTPG == 9)
             {
                   newphi = 2;
             }
           else if (phiTPG == 13)
             {
                   newphi = 3;
             }
           else if (phiTPG == 17)
             {
                   newphi = 4;
             }
           else if (phiTPG == 21)
             {
                   newphi = 5;
             }
           else if (phiTPG == 25)
             {
                   newphi = 6;
             }
           else if (phiTPG == 29)
             {
                   newphi = 7;
             }
           else if (phiTPG == 33)
             {
                   newphi = 8;
             }
           else if (phiTPG == 37)
             {
                   newphi = 9;
             }
           else if (phiTPG == 41)
             {
                   newphi = 10;
             }
           else if (phiTPG == 45)
             {
                   newphi = 11;
             }
           else if (phiTPG == 49)
             {
                   newphi = 12;
             }
           else if (phiTPG == 53)
              {
                   newphi = 13;
              }
           else if (phiTPG == 57)
              {
                       newphi = 14;
             }
           else if (phiTPG == 61)
              {
                   newphi = 15;
              }
           else if (phiTPG == 65)
              {
                   newphi = 16;
              }
           else if (phiTPG == 69)
              {
                   newphi = 17;
              }
                  return newphi;
}
void HcalTriggerPrimitiveAlgo::run(const HcalTPGCoder* incoder,
                                   const HcalTPGCompressor* outcoder,
                                   const HBHEDigiCollection& hbheDigis,
                                   const HFDigiCollection& hfDigis,
                                   HcalTrigPrimDigiCollection& result,
				   const HcalTrigTowerGeometry* trigTowerGeometry,
                                   float rctlsb) {
   theTrigTowerGeometry = trigTowerGeometry;
    
   incoder_=dynamic_cast<const HcaluLUTTPGCoder*>(incoder);
   outcoder_=outcoder;

   theSumMap.clear();
   theTowerMapFGSum.clear();
   HF_Veto.clear();
   fgMap_.clear();

   // do the HB/HE digis
   for(HBHEDigiCollection::const_iterator hbheItr = hbheDigis.begin();
   hbheItr != hbheDigis.end(); ++hbheItr) {
      addSignal(*hbheItr);
   }

   // and the HF digis
   for(HFDigiCollection::const_iterator hfItr = hfDigis.begin();
   hfItr != hfDigis.end(); ++hfItr) {
      addSignal(*hfItr);

   }

   int phiBin = 18;
   int etaBin = 22;
   int thresholds = 2;

   int fiberQIEThresh[phiBin][etaBin][thresholds];
   int FGBit[18][22];

   for (int i = 0; i < phiBin; i++)
   {
     for (int j = 0; j < etaBin; j++)
     {
       FGBit[i][j] = 0;
       for (int k = 0; k < thresholds; k++)
       {
         fiberQIEThresh[i][j][k] = 0;
       }
     }
   }

   for(HFDigiCollection::const_iterator hfdigi = hfDigis.begin(); hfdigi != hfDigis.end(); hfdigi++)
   {
     HcalDetId hcalid = HcalDetId(hfdigi->id()) ;

     int ieta = hcalid.ieta();
     int iphi = hcalid.iphi();
     int presample = hfdigi->presamples();
 
     double qieval = hfdigi->sample(presample).adc();
 
     int GCTeta = converttoGCTeta(ieta);
     int GCTphi = converttoGCTphi(iphi);

     if (qieval > FG_threshold1_ )
     {
       fiberQIEThresh[GCTphi][GCTeta][0]++;
     }
     if (qieval > FG_threshold2_)
     {
       fiberQIEThresh[GCTphi][GCTeta][1]++; 
     }
   }

   // Calculate the final FG bit
   for (int i = 0; i < phiBin; i++)
   {
     for (int j = 0; j < etaBin; j++)
     {
       if (fiberQIEThresh[i][j][0] > 0 && j < 4)
       {
         FGBit[i][0] = 1;
         FGBit[i][2] = 1;
       }
       if (fiberQIEThresh[i][j][0] > 0 && j > 17)
       {
         FGBit[i][19] = 1;
         FGBit[i][21] = 1;
       }
       if (fiberQIEThresh[i][j][1] > 0 && j < 4)
       {
         FGBit[i][1] = 1;
         FGBit[i][3] = 1;
       }
       if (fiberQIEThresh[i][j][1] > 0 && j > 17)
       {
         FGBit[i][18] = 1;
         FGBit[i][20] = 1;
       }
     }
   }

 
  for(SumMap::iterator mapItr = theSumMap.begin(); mapItr != theSumMap.end(); ++mapItr) {
      result.push_back(HcalTriggerPrimitiveDigi(mapItr->first));
      HcalTrigTowerDetId detId(mapItr->second.id());
      if(detId.ietaAbs() >= theTrigTowerGeometry->firstHFTower())
      { 
        int ietaTow = detId.ieta();
	int iphiTow = detId.iphi();
	analyzeHF(ietaTow, iphiTow, FGBit, mapItr->second, result.back(), rctlsb);
      }
      else{ analyze(mapItr->second, result.back()); }
   }	 	
   return;
}


void HcalTriggerPrimitiveAlgo::addSignal(const HBHEDataFrame & frame) {
   //Hack for 300_pre10, should be removed.
   if (frame.id().depth()==5) return;

   std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(frame.id());
   assert(ids.size() == 1 || ids.size() == 2);
   IntegerCaloSamples samples1(ids[0], int(frame.size()));

   samples1.setPresamples(frame.presamples());
   incoder_->adc2Linear(frame, samples1);

   std::vector<bool> msb;
   incoder_->lookupMSB(frame, msb);

   if(ids.size() == 2) {
      // make a second trigprim for the other one, and split the energy
      IntegerCaloSamples samples2(ids[1], samples1.size());
      for(int i = 0; i < samples1.size(); ++i) {
         samples1[i] = uint32_t(samples1[i]*0.5);
         samples2[i] = samples1[i];
      }
      samples2.setPresamples(frame.presamples());
      addSignal(samples2);
      addFG(ids[1], msb);
   }
   addSignal(samples1);
   addFG(ids[0], msb);
}


void HcalTriggerPrimitiveAlgo::addSignal(const HFDataFrame & frame) {

   if(frame.id().depth() == 1 || frame.id().depth() == 2) {
      std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(frame.id());
      assert(ids.size() == 1);
      IntegerCaloSamples samples(ids[0], frame.size());
      samples.setPresamples(frame.presamples());
      incoder_->adc2Linear(frame, samples);

      // Don't add to final collection yet
      // HF PMT veto sum is calculated in analyzerHF()
      IntegerCaloSamples zero_samples(ids[0], frame.size());
      zero_samples.setPresamples(frame.presamples());
      addSignal(zero_samples);

      // Mask off depths: fgid is the same for both depths
      uint32_t fgid = (frame.id().maskDepth());

      if ( theTowerMapFGSum.find(ids[0]) == theTowerMapFGSum.end() ) {
         SumFGContainer sumFG;
         theTowerMapFGSum.insert(std::pair<HcalTrigTowerDetId, SumFGContainer >(ids[0], sumFG));
      }

      SumFGContainer& sumFG = theTowerMapFGSum[ids[0]];
      SumFGContainer::iterator sumFGItr;
      for ( sumFGItr = sumFG.begin(); sumFGItr != sumFG.end(); ++sumFGItr) {
         if (sumFGItr->id() == fgid) break;
      }
      // If find
      if (sumFGItr != sumFG.end()) {
         for (int i=0; i<samples.size(); ++i) (*sumFGItr)[i] += samples[i];
      }
      else {
         //Copy samples (change to fgid)
         IntegerCaloSamples sumFGSamples(DetId(fgid), samples.size());
         sumFGSamples.setPresamples(samples.presamples());
         for (int i=0; i<samples.size(); ++i) sumFGSamples[i] = samples[i];
         sumFG.push_back(sumFGSamples);
      }

      // set veto to true if Long or Short less than threshold
      if (HF_Veto.find(fgid) == HF_Veto.end()) {
         vector<bool> vetoBits(samples.size(), false);
         HF_Veto[fgid] = vetoBits;
      }
      for (int i=0; i<samples.size(); ++i)
         if (samples[i] < minSignalThreshold_)
            HF_Veto[fgid][i] = true;
   }
}


void HcalTriggerPrimitiveAlgo::addSignal(const IntegerCaloSamples & samples) {
   HcalTrigTowerDetId id(samples.id());
   SumMap::iterator itr = theSumMap.find(id);
   if(itr == theSumMap.end()) {
      theSumMap.insert(std::make_pair(id, samples));
   }
   else {
      // wish CaloSamples had a +=
      for(int i = 0; i < samples.size(); ++i) {
         (itr->second)[i] += samples[i];
      }
   }
}


void HcalTriggerPrimitiveAlgo::analyze(IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result) {
   int shrink = weights_.size() - 1;
   std::vector<bool>& msb = fgMap_[samples.id()];
   IntegerCaloSamples sum(samples.id(), samples.size());

   //slide algo window
   for(int ibin = 0; ibin < int(samples.size())- shrink; ++ibin) {
      int algosumvalue = 0;
      for(unsigned int i = 0; i < weights_.size(); i++) {
         //add up value * scale factor
         algosumvalue += int(samples[ibin+i] * weights_[i]);
      }
      if (algosumvalue<0) sum[ibin]=0;            // low-side
                                                  //high-side
      //else if (algosumvalue>0x3FF) sum[ibin]=0x3FF;
      else sum[ibin] = algosumvalue;              //assign value to sum[]
   }

   // Align digis and TP
   int dgPresamples=samples.presamples(); 
   int tpPresamples=numberOfPresamples_;
   int shift = dgPresamples - tpPresamples;
   int dgSamples=samples.size();
   int tpSamples=numberOfSamples_;
   if(peakfind_){
       if((shift<shrink) || (shift + tpSamples + shrink > dgSamples - (peak_finder_algorithm_ - 1) )   ){
	    edm::LogInfo("HcalTriggerPrimitiveAlgo::analyze") << 
		"TP presample or size from the configuration file is out of the accessible range. Using digi values from data instead...";
	    shift=shrink;
	    tpPresamples=dgPresamples-shrink;
	    tpSamples=dgSamples-(peak_finder_algorithm_-1)-shrink-shift;
       }
   }

   std::vector<bool> finegrain(tpSamples,false);

   IntegerCaloSamples output(samples.id(), tpSamples);
   output.setPresamples(tpPresamples);

   for (int ibin = 0; ibin < tpSamples; ++ibin) {
      // ibin - index for output TP
      // idx - index for samples + shift
      int idx = ibin + shift;

      //Peak finding
      if (peakfind_) {
         bool isPeak = false;
         switch (peak_finder_algorithm_) {
            case 1 :
               isPeak = (samples[idx] > samples[idx-1] && samples[idx] >= samples[idx+1] && samples[idx] > theThreshold);
               break;
            case 2:
               isPeak = (sum[idx] > sum[idx-1] && sum[idx] >= sum[idx+1] && sum[idx] > theThreshold);
               break;
            default:
               break;
         }

         if (isPeak){
            output[ibin] = std::min<unsigned int>(sum[idx],0x3FF);
            finegrain[ibin] = msb[idx];
         }
         // Not a peak
         else output[ibin] = 0;
      }
      else { // No peak finding, just output running sum
         output[ibin] = std::min<unsigned int>(sum[idx],0x3FF);
         finegrain[ibin] = msb[idx];
      }

      // Only Pegged for 1-TS algo.
      if (peak_finder_algorithm_ == 1) {
         if (samples[idx] >= 0x3FF)
            output[ibin] = 0x3FF;
      }
   }
   outcoder_->compress(output, finegrain, result);
}


void HcalTriggerPrimitiveAlgo::analyzeHF(int ieta, int iphi, int (&FGBit)[18][22], IntegerCaloSamples & samples, HcalTriggerPrimitiveDigi & result, float rctlsb) {
   HcalTrigTowerDetId detId(samples.id());

   // Align digis and TP
   int dgPresamples=samples.presamples(); 
   int tpPresamples=numberOfPresamplesHF_;
   int shift = dgPresamples - tpPresamples;
   int dgSamples=samples.size();
   int tpSamples=numberOfSamplesHF_;
   if(shift<0 || shift+tpSamples>dgSamples){
	edm::LogInfo("HcalTriggerPrimitiveAlgo::analyzeHF") << 
	    "TP presample or size from the configuration file is out of the accessible range. Using digi values from data instead...";
	tpPresamples=dgPresamples;
	shift=0;
	tpSamples=dgSamples;
   }
   
   std::vector<bool> finegrain(tpSamples, false);

   int ietaGCT = converttoGCTetaFromTPG(ieta);
   int iphiGCT = converttoGCTphiFromTPG(iphi); 
 
   for (int index = 0; index < tpSamples; ++index)
   {
     finegrain[index] = FGBit[iphiGCT][ietaGCT];
   }

   TowerMapFGSum::const_iterator tower2fg = theTowerMapFGSum.find(detId);
   assert(tower2fg != theTowerMapFGSum.end());

   const SumFGContainer& sumFG = tower2fg->second;
   // Loop over all L+S pairs that mapped from samples.id()
   // Note: 1 samples.id() = 6 x (L+S) without noZS
   for (SumFGContainer::const_iterator sumFGItr = sumFG.begin(); sumFGItr != sumFG.end(); ++sumFGItr) {
      const std::vector<bool>& veto = HF_Veto[sumFGItr->id().rawId()];
    for (int ibin = 0; ibin < tpSamples; ++ibin) {
         int idx = ibin + shift;
         // if not vetod, add L+S to total sum and calculate FG
	 bool vetoed = idx<int(veto.size()) && veto[idx];
         if (!(vetoed && (*sumFGItr)[idx] > PMT_NoiseThreshold_)) {
            samples[idx] += (*sumFGItr)[idx];
	 }
      }
   }

   IntegerCaloSamples output(samples.id(), tpSamples);
   output.setPresamples(tpPresamples);

   for (int ibin = 0; ibin < tpSamples; ++ibin) {
      int idx = ibin + shift;
      output[ibin] = samples[idx] / (rctlsb == 0.25 ? 4 : 8);
      if (output[ibin] > 0x3FF) output[ibin] = 0x3FF;
   }
   outcoder_->compress(output, finegrain, result);
}

void HcalTriggerPrimitiveAlgo::runZS(HcalTrigPrimDigiCollection & result){
   for (HcalTrigPrimDigiCollection::iterator tp = result.begin(); tp != result.end(); ++tp){
      bool ZS = true;
      for (int i=0; i<tp->size(); ++i) {
         if (tp->sample(i).compressedEt()  > ZS_threshold_I_) {
            ZS=false;
            break;
         }
      }
      if (ZS) tp->setZSInfo(false,true);
      else tp->setZSInfo(true,false);
   }
}

void HcalTriggerPrimitiveAlgo::runFEFormatError(const FEDRawDataCollection* rawraw,
                                                const HcalElectronicsMap *emap,
                                                HcalTrigPrimDigiCollection & result
                                                ){
  std::set<uint32_t> FrontEndErrors;

  for(int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; ++i) {
    const FEDRawData& raw = rawraw->FEDData(i);
    if (raw.size()<12) continue;
    const HcalDCCHeader* dccHeader=(const HcalDCCHeader*)(raw.data());
    if(!dccHeader) continue;
    HcalHTRData htr;
    for (int spigot=0; spigot<HcalDCCHeader::SPIGOT_COUNT; spigot++) {
      if (!dccHeader->getSpigotPresent(spigot)) continue;
      dccHeader->getSpigotData(spigot,htr,raw.size());
      int dccid = dccHeader->getSourceId();
      int errWord = htr.getErrorsWord() & 0x1FFFF;
      bool HTRError = (!htr.check() || htr.isHistogramEvent() || (errWord & 0x800)!=0);

      if(HTRError) {
        bool valid =false;
        for(int fchan=0; fchan<3 && !valid; fchan++) {
          for(int fib=0; fib<9 && !valid; fib++) {
            HcalElectronicsId eid(fchan,fib,spigot,dccid-FEDNumbering::MINHCALFEDID);
            eid.setHTR(htr.readoutVMECrateId(),htr.htrSlot(),htr.htrTopBottom());
            DetId detId = emap->lookup(eid);
            if(detId.null()) continue;
            HcalSubdetector subdet=(HcalSubdetector(detId.subdetId()));
            if (detId.det()!=4||
              (subdet!=HcalBarrel && subdet!=HcalEndcap &&
              subdet!=HcalForward )) continue;
            std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry->towerIds(detId);
            for (std::vector<HcalTrigTowerDetId>::const_iterator triggerId=ids.begin(); triggerId != ids.end(); ++triggerId) {
              FrontEndErrors.insert(triggerId->rawId());
            }
            //valid = true;
          }
        }
      }
    }
  }

  // Loop over TP collection
  // Set TP to zero if there is FE Format Error
  HcalTriggerPrimitiveSample zeroSample(0);
  for (HcalTrigPrimDigiCollection::iterator tp = result.begin(); tp != result.end(); ++tp){
    if (FrontEndErrors.find(tp->id().rawId()) != FrontEndErrors.end()) {
      for (int i=0; i<tp->size(); ++i) tp->setSample(i, zeroSample);
    }
  }
}

void HcalTriggerPrimitiveAlgo::addFG(const HcalTrigTowerDetId& id, std::vector<bool>& msb){
   FGbitMap::iterator itr = fgMap_.find(id);
   if (itr != fgMap_.end()){
      std::vector<bool>& _msb = itr->second;
      for (size_t i=0; i<msb.size(); ++i)
         _msb[i] = _msb[i] || msb[i];
   }
   else fgMap_[id] = msb;
}

void HcalTriggerPrimitiveAlgo::setPeakFinderAlgorithm(int algo){
   if (algo <=0 && algo>2)
      throw cms::Exception("ERROR: Only algo 1 & 2 are supported.") << std::endl;
   peak_finder_algorithm_ = algo;
}
