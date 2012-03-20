#include <iostream>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalUpgradeTriggerPrimitiveAlgo.h"

HcalUpgradeTriggerPrimitiveAlgo::HcalUpgradeTriggerPrimitiveAlgo(bool pf, const std::vector<double>& w, int latency, 
							   uint32_t FGThreshold, uint32_t ZSThreshold, uint32_t MinSignalThreshold,  uint32_t PMTNoiseThreshold,
							   int numberOfSamples, int numberOfPresamples, bool excludeDepth5 ):
  incoder_(0), 
  outcoder_(0), 
  
  peakfind_(pf), 
  peak_finder_algorithm_ ( 2 ) ,
  weights_(w), 
  latency_(latency), 

  thePFThreshold_(0),
  theFGThreshold_(FGThreshold),
  theZSThreshold_(ZSThreshold),
  theMinSignalThreshold_(MinSignalThreshold),
  thePMTNoiseThreshold_(PMTNoiseThreshold),

  numberOfSamples_(numberOfSamples),
  numberOfPresamples_(numberOfPresamples),
  
  excludeDepth5_(excludeDepth5)
{}
  
HcalUpgradeTriggerPrimitiveAlgo::~HcalUpgradeTriggerPrimitiveAlgo(){}

void HcalUpgradeTriggerPrimitiveAlgo::run(const HcalTPGCoder * incoder,
				       const HcalTPGCompressor * outcoder,
				       const HBHEDigiCollection & hbheDigis,
				       const HFDigiCollection & hfDigis,
				       HcalUpgradeTrigPrimDigiCollection & result){
  
  //------------------------------------------------------
  // Set coders and clear energy maps
  //------------------------------------------------------
  
  incoder_=dynamic_cast<const HcaluLUTTPGCoder*>(incoder);
  outcoder_=outcoder;
  
  theSumMap.clear();
  
  //------------------------------------------------------
  // Loop over the digi collections and fill energy maps
  //------------------------------------------------------
  
  HBHEDigiCollection::const_iterator hbheItr     = hbheDigis.begin();
  HBHEDigiCollection::const_iterator hbheItr_end = hbheDigis.end();
  
  HFDigiCollection::const_iterator   hfItr       = hfDigis.begin();
  HFDigiCollection::const_iterator   hfItr_end   = hfDigis.end();
  
  for(; hbheItr != hbheItr_end; ++hbheItr)
    addSignal(*hbheItr);

  for(; hfItr != hfItr_end; ++hfItr)
    addSignal(*hfItr);

  //------------------------------------------------------
  // Once energy maps are filled,
  //   1 - Loop over entries in the energy sum map
  //   2 - Convert summed IntegerCaloSamples to TP's
  //------------------------------------------------------
  
  for(SumMap::iterator mapItr = theSumMap.begin(); mapItr != theSumMap.end(); ++mapItr){
      
    // Push back an empty TP digi that contains only the HcalTrigTowerDetId
    result.push_back(HcalUpgradeTriggerPrimitiveDigi(mapItr->first));
    
    HcalTrigTowerDetId detId((mapItr->second).id());
    
    // Fill in the rest of the information for the TP digi
    if(detId.ietaAbs() >= theTrigTowerGeometry.firstHFTower())
      analyzeHF(mapItr->second, result.back());
    else
      analyze  (mapItr->second, result.back());
  }

  //------------------------------------------------------
  // Free up memory and return
  //------------------------------------------------------

  theSumMap.clear();  

  return;
}

//------------------------------------------------------
// Compression function
//------------------------------------------------------

void HcalUpgradeTriggerPrimitiveAlgo::fillDepth1Frame ( const HBHEDataFrame & frame, HBHEDataFrame & depth1_frame ){

  int size = frame.size();
  int pres = frame.presamples();
  
   depth1_frame.setSize       ( size );
   depth1_frame.setPresamples ( pres );

   for ( int isample = 0; isample < size; ++isample)
      depth1_frame.setSample(isample,frame[isample]);

}
/*
void HcalUpgradeTriggerPrimitiveAlgo::getRightDepthCaloSample ( const HcalDetId & hcaldetId,
				  const IntegerCaloSamples& depth1_sample,
				  IntegerCaloSamples & sample );
*/

void HcalUpgradeTriggerPrimitiveAlgo::adc2Linear(const HBHEDataFrame& frame, IntegerCaloSamples & sample ){
  
   
   bool detIdIsValid = false;

   // int ieta = frame.id().ieta();
   // int iphi = frame.id().iphi();
   int depth = frame.id().depth();
   
   while ( !detIdIsValid ){
      
      detIdIsValid = HcalDetId::validDetId ( frame.id().subdet(),
					     frame.id().ieta(),
					     frame.id().iphi(),
					     depth );
      
      if (!detIdIsValid) depth--;
   }
   
   HcalDetId * depth1_frame_id  = new HcalDetId ( frame.id().subdet(),
						  frame.id().ieta(),
						  frame.id().iphi(),
						  depth );
   
   HBHEDataFrame * depth1_frame
      = new HBHEDataFrame( * depth1_frame_id );

   fillDepth1Frame        ( frame, * depth1_frame );
   incoder_ -> adc2Linear ( * depth1_frame, sample );
  
   if (depth1_frame_id) delete depth1_frame_id;
   if (depth1_frame) delete depth1_frame;

}

void HcalUpgradeTriggerPrimitiveAlgo::fillRightDepthSamples ( 
   const IntegerCaloSamples & depth1_sample,
   IntegerCaloSamples & sample ){

   int size = depth1_sample.size();

   for (int isample = 0; isample < size; ++isample)
      sample[isample] = depth1_sample[isample];

}

//------------------------------------------------------
// Add signals from the HBHE digis to the mapping
// 
// This should be the same as the pre-upgrade method.
//------------------------------------------------------

void HcalUpgradeTriggerPrimitiveAlgo::addSignal(const HBHEDataFrame & frame) {

  //------------------------------------------------------
  // "Hack for 300_pre10" from the original code.
  // User can turn this off in the python cfg file.
  //------------------------------------------------------
  
  if (excludeDepth5_ && frame.id().depth()==5) return;

  //------------------------------------------------------
  // Get the trigger tower id(s) for this digi
  //------------------------------------------------------
  
  std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
  assert(ids.size() == 1 || ids.size() == 2);

  IntegerCaloSamples samples1(ids[0], int(frame.size()));
  IntegerCaloSamples samples1_depth1(ids[0], int(frame.size()));

  samples1.setPresamples(frame.presamples());

  //------------------------------------------------------
  // Compress HBHEDataFrame ADC -> IntegerCaloSamples
  //------------------------------------------------------
  
  adc2Linear( frame, samples1 );
  
  //------------------------------------------------------
  // If there are two id's make a second trigprim for 
  // the other one, and split the energy
  //------------------------------------------------------

  if(ids.size() == 2) {
    
    IntegerCaloSamples samples2(ids[1], samples1.size());

    for(int i = 0; i < samples1.size(); ++i) {
      samples1[i] = uint32_t(samples1[i]*0.5);
      samples2[i] = samples1[i];
    }
    
    samples2.setPresamples(frame.presamples());
    addSignal(samples2);
  }
  
  addSignal(samples1);

}


void HcalUpgradeTriggerPrimitiveAlgo::addSignal(const HFDataFrame & frame) {
  
  int depth = frame.id().depth();
  
  if( depth == 1 || depth == 2) {

    std::vector<HcalTrigTowerDetId> ids = theTrigTowerGeometry.towerIds(frame.id());
    
    assert(ids.size() == 1);
    
    IntegerCaloSamples samples(ids[0], frame.size());
    
    samples.setPresamples(frame.presamples());

    incoder_->adc2Linear(frame, samples);
    
    addSignal(samples);

  }
}
  
//------------------------------------------------------
// Add this IntegerCaloSamples to SumMap
//------------------------------------------------------

void HcalUpgradeTriggerPrimitiveAlgo::addSignal(const IntegerCaloSamples & samples ){

  //------------------------------------------------------
  // If this ID isn't in the map already, add it
  //------------------------------------------------------

  HcalTrigTowerDetId id(samples.id());
  SumMap::iterator itr = theSumMap.find(id);
  if(itr == theSumMap.end()) {
    theSumMap.insert ( std::pair <HcalTrigTowerDetId, IntegerCaloSamples>( id , samples) );
  } 

  //------------------------------------------------------
  // If this ID is in the map, add to the existing entry
  //------------------------------------------------------
  
  else {
    for(int i = 0; i < samples.size(); ++i)  {
      (itr->second)[i] += samples[i];   
    }
  }
}

//------------------------------------------------------
// Go from an IntegerCaloSamples to a TP digi (for HBHE)
//------------------------------------------------------

void HcalUpgradeTriggerPrimitiveAlgo::analyze(IntegerCaloSamples & samples, 
					   HcalUpgradeTriggerPrimitiveDigi & result){

  //------------------------------------------------------
  // First find output samples length and new presamples
  //------------------------------------------------------
  
  int shrink       = weights_.size()-1; 
  int outlength    = samples.size() - shrink;
  
  //------------------------------------------------------
  // Declare all IntegerCaloSamples required
  // NOTE: this is a really inefficient way of doing this.  
  // Must be fixed.
  //------------------------------------------------------

  IntegerCaloSamples allSum       (samples.id(), samples.size());
  IntegerCaloSamples allCollapsed (samples.id(), numberOfSamples_  );  

  std::vector<int> nullFineGrain ( allCollapsed.size(), 0 );

  //------------------------------------------------------
  // Sum over all samples and make a summed sample
  // 
  // doSampleSum returns a bool that says if the input
  //   IntegerCaloSample is saturated
  //------------------------------------------------------
  
  bool allSOI_pegged = doSampleSum (samples , allSum, outlength);

  //------------------------------------------------------
  // Collapse the sample.  Peakfinder is here.
  //------------------------------------------------------
    
  doSampleCollapse (samples , allSum, allCollapsed);
  
  //------------------------------------------------------
  // Compress into an HcalUpgradeTriggerPrimitiveDigi
  // FineGrain is left blank intentionally -- this should 
  // be filled in later.
  //------------------------------------------------------
  
  doSampleCompress ( allCollapsed, nullFineGrain, result );
  
}

//------------------------------------------------------
// Go from an IntegerCaloSamples to a TP digi (for HF)
//------------------------------------------------------

void HcalUpgradeTriggerPrimitiveAlgo::analyzeHF(IntegerCaloSamples & samples, 
					     HcalUpgradeTriggerPrimitiveDigi & result){
  
  //------------------------------------------------------
  // Declare needed IntegerCaloSamples 
  //------------------------------------------------------
  
  IntegerCaloSamples output(samples.id(),numberOfSamples_);
  output.setPresamples(numberOfPresamples_);

  //------------------------------------------------------
  // "Collapse" algorithm for HF is just a scaling
  // Saturation is at 10 bits    
  //------------------------------------------------------

  int shift = samples.presamples() - numberOfPresamples_;
  
  for (int ibin = 0; ibin < numberOfSamples_; ++ibin) {
    int idx = ibin + shift;
    output[ibin] = samples[idx] / 4;
    if (output[ibin] > 0x3FF) output[ibin] = 0x3FF;
  }
  
  //------------------------------------------------------
  // Get null fine grain
  //------------------------------------------------------

  std::vector<int>   nullFineGrain ( samples.size(), 0 );
  
  doSampleCompress ( output,  nullFineGrain, result );  
}

//------------------------------------------------------
// "Collapse" method for HBHE.  Peakfinder is here.
//------------------------------------------------------

void HcalUpgradeTriggerPrimitiveAlgo::doSampleCollapse (const IntegerCaloSamples& originalSamples,
						     const IntegerCaloSamples& summedSamples,
						     IntegerCaloSamples& collapsedSamples  ){
  
  //------------------------------------------------------
  // Calculate presample shift
  //------------------------------------------------------

  int shift = originalSamples.presamples() - numberOfPresamples_;

  //------------------------------------------------------
  // Do peak finding
  //------------------------------------------------------

  for (int ibin = 0; ibin < numberOfSamples_ ; ++ibin ){ 

    int idx = ibin + shift;

    if(peakfind_) {

      bool isPeak = false;
      
      switch (peak_finder_algorithm_) {
	
      case 1 :
	isPeak = (originalSamples[idx] >  originalSamples[idx-1] && 
		  originalSamples[idx] >= originalSamples[idx+1] && 
		  originalSamples[idx] >  thePFThreshold_ );
	break;
	
      case 2:
	isPeak = (summedSamples  [idx] >  summedSamples  [idx-1] && 
		  summedSamples  [idx] >= summedSamples  [idx+1] && 
		  summedSamples  [idx] >  thePFThreshold_ );
	break;

      default:
	break;
      }

      if ( isPeak ) {
	collapsedSamples[ibin] = std::min <unsigned int> ( summedSamples[idx], 0x3FF );
	// fine grain
      }
      else collapsedSamples[ibin] = 0;

    }

    else { // No peakfinding
      collapsedSamples[ibin] = std::min <unsigned int> ( summedSamples[idx], 0x3FF );
      // fine grain
    }

    // Only Pegged for 1-TS algo.
    if (peak_finder_algorithm_ == 1) {
      if (originalSamples[idx] >= 0x3FF)
	collapsedSamples[ibin] = 0x3FF;
    }
  }  
}

//------------------------------------------------------
// Weighted sum method
//------------------------------------------------------

bool HcalUpgradeTriggerPrimitiveAlgo::doSampleSum (const IntegerCaloSamples& inputSamples, 
						IntegerCaloSamples& summedSamples,
						int outlength ){
  
  
  bool SOI_pegged = (inputSamples[inputSamples.presamples()] > 0x3FF);
  
  //slide algo window
  for(int ibin = 0; ibin < outlength ; ++ibin) {
    
    int algosumvalue = 0;
    
    //add up value * scale factor      
    for(unsigned int i = 0; i < weights_.size(); i++) 
      algosumvalue += int(inputSamples[ibin+i] * weights_[i]);
    
    if   (algosumvalue < 0) summedSamples[ibin] = 0;           // low-side
    else                    summedSamples[ibin] = algosumvalue;// assign value to sum[]
  }
  
  return SOI_pegged;

}

//------------------------------------------------------
// Compression method (for HBHE and HF)
//------------------------------------------------------

void HcalUpgradeTriggerPrimitiveAlgo::doSampleCompress (const IntegerCaloSamples& etSamples,
						     const std::vector<int> & fineGrainSamples,
						     HcalUpgradeTriggerPrimitiveDigi & digi){
  
  //------------------------------------------------------
  // Looks like we have to go through 
  //   HcalTriggerPrimitiveSample in order to compress.
  //------------------------------------------------------

  for (int i = 0; i < etSamples.size(); ++i){
    
    int compressedEt  = outcoder_ -> compress ( etSamples.id() , etSamples [i], false ).compressedEt();
    int fineGrain     = fineGrainSamples[i];

    digi.setSample( i, HcalUpgradeTriggerPrimitiveSample ( compressedEt, fineGrain, 0, 0 ));

  }

  digi.setPresamples ( numberOfPresamples_ );

}
