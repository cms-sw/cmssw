// File: SiStripDigitizerAlgorithm.cc
// Description:  Steering class for digitization.

#include <vector>
#include <algorithm>
#include <iostream>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimTracker/SiStripDigitizer/interface/SiStripDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "CLHEP/Random/RandFlat.h"

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng):
  conf_(conf),rndEngine(eng){
  theThreshold              = conf_.getParameter<double>("NoiseSigmaThreshold");
  theFedAlgo                = conf_.getParameter<int>("FedAlgorithm");
  peakMode                  = conf_.getParameter<bool>("APVpeakmode");
  theElectronPerADC         = conf_.getParameter<double>( peakMode ? "electronPerAdcPeak" : "electronPerAdcDec" );
  noise                     = conf_.getParameter<bool>("Noise");
  zeroSuppression           = conf_.getParameter<bool>("ZeroSuppression");
  theTOFCutForPeak          = conf_.getParameter<double>("TOFCutForPeak");
  theTOFCutForDeconvolution = conf_.getParameter<double>("TOFCutForDeconvolution");
  cosmicShift               = conf_.getUntrackedParameter<double>("CosmicDelayShift");
  inefficiency              = conf_.getParameter<double>("Inefficiency");
  RealPedestals             = conf_.getParameter<bool>("RealPedestals"); 
  SingleStripNoise          = conf_.getParameter<bool>("SingleStripNoise");
  CommonModeNoise           = conf_.getParameter<bool>("CommonModeNoise");
  BaselineShift             = conf_.getParameter<bool>("BaselineShift");
  APVSaturationFromHIP      = conf_.getParameter<bool>("APVSaturationFromHIP");
  APVSaturationProb         = conf_.getParameter<double>("APVSaturationProb");
  cmnRMStib                 = conf_.getParameter<double>("cmnRMStib");
  cmnRMStob                 = conf_.getParameter<double>("cmnRMStob");
  cmnRMStid                 = conf_.getParameter<double>("cmnRMStid");
  cmnRMStec                 = conf_.getParameter<double>("cmnRMStec");
  pedOffset                 = (unsigned int)conf_.getParameter<double>("PedestalsOffset");
  if (peakMode) {
    tofCut=theTOFCutForPeak;
    LogDebug("StripDigiInfo")<<"APVs running in peak mode (poor time resolution)";
  } else {
    tofCut=theTOFCutForDeconvolution;
    LogDebug("StripDigiInfo")<<"APVs running in deconvolution mode (good time resolution)";
  };
  
  theSiHitDigitizer = new SiHitDigitizer(conf_,rndEngine);
  theSiPileUpSignals = new SiPileUpSignals();
  theSiNoiseAdder = new SiGaussianTailNoiseAdder(theThreshold,rndEngine);
  theSiDigitalConverter = new SiTrivialDigitalConverter(theElectronPerADC);
  theSiZeroSuppress = new SiStripFedZeroSuppression(theFedAlgo);
  theFlatDistribution = new CLHEP::RandFlat(rndEngine, 0., 1.);    

  if(inefficiency == 20) {  // simulate 20% (random) data loss in TIB 1 and 2
    theStripInefficiency_[0] = 0.2;
    theStripInefficiency_[1] = 0.2;
    theStripInefficiency_[2] = 0.;
    theStripInefficiency_[3] = 0.;
  } else if(inefficiency == 30) {  // simulate 50% (random) data loss in TIB 1 and 2
    theStripInefficiency_[0] = 0.5;
    theStripInefficiency_[1] = 0.5;
    theStripInefficiency_[2] = 0.;
    theStripInefficiency_[3] = 0.;
  } else if(inefficiency == 40) {  // simulate 99% (random) data loss in TIB 1 and 2
    theStripInefficiency_[0] = 0.99;
    theStripInefficiency_[1] = 0.99;
    theStripInefficiency_[2] = 0.;
    theStripInefficiency_[3] = 0.;
  } else {
    theStripInefficiency_[0] = inefficiency;
    theStripInefficiency_[1] = inefficiency;
    theStripInefficiency_[2] = inefficiency;
    theStripInefficiency_[3] = inefficiency;
  }
  edm::LogInfo("SiStripDigitizer")<< " strip digis setup with inefficiency = " << inefficiency 
            << " theStripInefficiency_ = " << theStripInefficiency_[0] << " "
            << " " << theStripInefficiency_[1] << " "
            << " " << theStripInefficiency_[2] << " "
            << " " << theStripInefficiency_[3];  
}

SiStripDigitizerAlgorithm::~SiStripDigitizerAlgorithm(){
  delete theSiHitDigitizer;
  delete theSiPileUpSignals;
  delete theSiNoiseAdder;
  delete theSiDigitalConverter;
  delete theSiZeroSuppress;
  delete theFlatDistribution;
  //delete particle;
  //delete pdt;
}

//  Run the algorithm for a given module
//  ------------------------------------

void SiStripDigitizerAlgorithm::run(edm::DetSet<SiStripDigi>& outdigi,
				    edm::DetSet<SiStripRawDigi>& outrawdigi,
				    const std::vector<std::pair<const PSimHit*, int > > &input,
				    StripGeomDetUnit *det,
				    GlobalVector bfield,float langle, 
				    edm::ESHandle<SiStripGain> & gainHandle,
				    edm::ESHandle<SiStripThreshold> & thresholdHandle,
				    edm::ESHandle<SiStripNoises> & noiseHandle,
				    edm::ESHandle<SiStripPedestals> & pedestalHandle,
					edm::ESHandle<SiStripBadStrip> & deadChannelHandle
				   ) {  
  theSiPileUpSignals->reset();
  unsigned int detID = det->geographicalId().rawId();
  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  SiStripApvGain::Range detGainRange = gainHandle->getRange(detID);
  SiStripPedestals::Range detPedestalRange = pedestalHandle->getRange(detID);
  SiStripBadStrip::Range detBadStripRange = deadChannelHandle->getRange(detID);
  numStrips = (det->specificTopology()).nstrips();  
  
  //stroing the bad stip of the the module. the module is not removed but just signal put to 0
  std::vector<bool> badChannels;
  badChannels.clear();
  badChannels.insert(badChannels.begin(),numStrips,false);
  SiStripBadStrip::data fs;
  for(SiStripBadStrip::ContainerIterator it=detBadStripRange.first;it!=detBadStripRange.second;++it){
  	fs=deadChannelHandle->decode(*it);
    for(int strip = fs.firstStrip; strip <fs.firstStrip+fs.range; ++strip )badChannels[strip] = true;
  }

     
  // local amplitude of detector channels (from processed PSimHit)
//  locAmpl.clear();
  detAmpl.clear();
//  locAmpl.insert(locAmpl.begin(),numStrips,0.);
  // total amplitude of detector channels
  detAmpl.insert(detAmpl.begin(),numStrips,0.);

  firstChannelWithSignal = numStrips;
  lastChannelWithSignal  = 0;

  // First: loop on the SimHits
  std::vector<std::pair<const PSimHit*, int > >::const_iterator simHitIter = input.begin();
  std::vector<std::pair<const PSimHit*, int > >::const_iterator simHitIterEnd = input.end();

  // simulate inefficiency simhit by simhit, so throw random number for each simhit if needed
  // default is no inefficiency simulation
  unsigned int subdetId = det->geographicalId().subdetId();
  bool do_inefficiency = false;
  float inefficiency2use = 0.;
  if(inefficiency == 20 || inefficiency == 30 || inefficiency == 40) {
    if (subdetId == StripSubdetector::TIB) { // special case for TIB-only inefficency
      TIBDetId tibid(det->geographicalId().rawId());
      int layerNumber = tibid.layer();
      inefficiency2use = theStripInefficiency_[layerNumber-1];
      if(inefficiency2use > 0) do_inefficiency = true;
    }
  } else if(inefficiency > 0. ) {  // global inefficiency for all strip layers
    do_inefficiency = true;
    inefficiency2use = inefficiency; // TODO cannot be right as inefficiency is an integer assume percent?
  }

  //if(theFlatDistribution->fire()>inefficiency) {
  for (;simHitIter != simHitIterEnd; ++simHitIter) {
    if(!do_inefficiency || (do_inefficiency && theFlatDistribution->fire()>inefficiency2use) ) {
      locAmpl.clear();
      locAmpl.insert(locAmpl.begin(),numStrips,0.);
      // check TOF
      if ( std::fabs( ((*simHitIter).first)->tof() - cosmicShift - det->surface().toGlobal(((*simHitIter).first)->localPosition()).mag()/30.) < tofCut && ((*simHitIter).first)->energyLoss()>0) {
        localFirstChannel = numStrips;
        localLastChannel  = 0;
        // process the hit
        theSiHitDigitizer->processHit(((*simHitIter).first),*det,bfield,langle, locAmpl, localFirstChannel, localLastChannel);
          
		  //APV Killer to simulate HIP effect
		  //------------------------------------------------------
		  
		  if(APVSaturationFromHIP&&!zeroSuppression){
		    int pdg_id = ((*simHitIter).first)->particleType();
			particle = pdt->particle(pdg_id);
			if(particle != NULL){
				float charge = particle->charge();
				bool isHadron = particle->isHadron();
			    if(charge!=0 && isHadron){
			  		if(theFlatDistribution->fire()<APVSaturationProb){
			 	   	 	int FirstAPV = localFirstChannel/128;
				 		int LastAPV = localLastChannel/128;
						std::cout << "-------------------HIP--------------" << std::endl;
						std::cout << "Killing APVs " << FirstAPV << " - " <<LastAPV << " " << detID <<std::endl;
				 		for(int strip = FirstAPV*128; strip < LastAPV*128 +128; ++strip) badChannels[strip] = true; //doing like that I remove the signal information only after the 
																													//stip that got the HIP but it remains the signal of the previous
																													//one. I'll make a further loop to remove all signal																										
						
			  		}
				}
			}
	      }             
		
		
		theSiPileUpSignals->add(locAmpl, localFirstChannel, localLastChannel, ((*simHitIter).first), (*simHitIter).second);
    
		// sum signal on strips
        for (size_t iChannel=localFirstChannel; iChannel<localLastChannel; iChannel++) {
          if(locAmpl[iChannel]>0.) {
		    //if(!badChannels[iChannel]) detAmpl[iChannel]+=locAmpl[iChannel];
			//locAmpl[iChannel]=0;
			detAmpl[iChannel]+=locAmpl[iChannel];
           }
        }
        if(firstChannelWithSignal>localFirstChannel) firstChannelWithSignal=localFirstChannel;
        if(lastChannelWithSignal<localLastChannel) lastChannelWithSignal=localLastChannel;
      }
	}
  }
  
  //removing signal from the dead (and HIP effected) strips
  for(int strip =0; strip < numStrips; ++strip) if(badChannels[strip]) detAmpl[strip] =0;
  
  const SiPileUpSignals::HitToDigisMapType& theLink(theSiPileUpSignals->dumpLink());  
  const SiPileUpSignals::HitCounterToDigisMapType& theCounterLink(theSiPileUpSignals->dumpCounterLink());  
  
  if(zeroSuppression){
    if(noise){
	  int RefStrip = int(numStrips/2.);
	  while(RefStrip<numStrips&&badChannels[RefStrip]){ //if the refstrip is bad, I move up to when I don't find it
	  	RefStrip++;
	  }
	  if(RefStrip<numStrips){
	 	float noiseRMS = noiseHandle->getNoise(RefStrip,detNoiseRange);
		float gainValue = gainHandle->getStripGain(RefStrip, detGainRange);
		theSiNoiseAdder->addNoise(detAmpl,firstChannelWithSignal,lastChannelWithSignal,numStrips,noiseRMS*theElectronPerADC/gainValue);
	  }
	}
    digis.clear();
    theSiZeroSuppress->suppress(theSiDigitalConverter->convert(detAmpl, gainHandle, detID), digis, detID,noiseHandle,thresholdHandle);
    push_link(digis, theLink, theCounterLink, detAmpl,detID);
    outdigi.data = digis;
  }
  
  
  if(!zeroSuppression){
    //if(noise){
      // the constant pedestal offset is needed because
      //   negative adc counts are not allowed in case
      //   Pedestal and CMN subtraction is performed.
      //   The pedestal value read from the conditions
      //   is pedValue and after the pedestal subtraction
      //   the baseline is zero. The Common Mode Noise
      //   is not subtracted from the negative adc counts
      //   channels. Adding pedOffset the baseline is set
      //   to pedOffset after pedestal subtraction and CMN
      //   is subtracted to all the channels since none of
      //   them has negative adc value. The pedOffset is
      //   treated as a constant component in the CMN
      //   estimation and subtracted as CMN.
      
         
		//calculating the charge deposited on each APV and subtracting the shift
		//------------------------------------------------------
		if(BaselineShift){
		   theSiNoiseAdder->addBaselineShift(detAmpl, badChannels);
		}
		
		//Adding the strip noise
		//------------------------------------------------------						 
		if(noise){
		    std::vector<float> noiseRMSv;
			noiseRMSv.clear();
		    noiseRMSv.insert(noiseRMSv.begin(),numStrips,0.);
			
		    if(SingleStripNoise){
			    for(int strip=0; strip< numStrips; ++strip){
			  		if(!badChannels[strip]) noiseRMSv[strip] = (noiseHandle->getNoise(strip,detNoiseRange))* theElectronPerADC;
			  	}
			
	    	} else {
			    int RefStrip = 0; //int(numStrips/2.);
		    	    while(RefStrip<numStrips&&badChannels[RefStrip]){ //if the refstrip is bad, I move up to when I don't find it
					RefStrip++;
				}
				if(RefStrip<numStrips){
					float noiseRMS = noiseHandle->getNoise(RefStrip,detNoiseRange) *theElectronPerADC;
					for(int strip=0; strip< numStrips; ++strip){
			       		if(!badChannels[strip]) noiseRMSv[strip] = noiseRMS;
			  		}
				}
			}
			
			theSiNoiseAdder->addNoiseVR(detAmpl, noiseRMSv);
		}			
		
		//adding the CMN
		//------------------------------------------------------
        if(CommonModeNoise){
		  float cmnRMS = 0.;
		  DetId  detId(detID);
		  uint32_t SubDet = detId.subdetId();
		  if(SubDet==3){
		    cmnRMS = cmnRMStib;
		  }else if(SubDet==4){
		    cmnRMS = cmnRMStid;
		  }else if(SubDet==5){
		    cmnRMS = cmnRMStob;
		  }else if(SubDet==6){
		    cmnRMS = cmnRMStec;
		  }
		  cmnRMS *= theElectronPerADC;
          theSiNoiseAdder->addCMNoise(detAmpl, cmnRMS, badChannels);
		}
		
        		
		//Adding the pedestals
		//------------------------------------------------------
		
		std::vector<float> vPeds;
		vPeds.clear();
		vPeds.insert(vPeds.begin(),numStrips,0.);
		
		if(RealPedestals){
		    for(int strip=0; strip< numStrips; ++strip){
			   if(!badChannels[strip]) vPeds[strip] = (pedestalHandle->getPed(strip,detPedestalRange)+pedOffset)* theElectronPerADC;
		    }
        } else {
		    for(int strip=0; strip< numStrips; ++strip){
			  if(!badChannels[strip]) vPeds[strip] = pedOffset* theElectronPerADC;
			}
		}
		
		theSiNoiseAdder->addPedestals(detAmpl, vPeds);	
		
		 
	//if(!RealPedestals&&!CommonModeNoise&&!noise&&!BaselineShift&&!APVSaturationFromHIP){
    //  edm::LogWarning("SiStripDigitizer")<<"You are running the digitizer without Noise generation and without applying Zero Suppression. ARE YOU SURE???";
    //}else{							 
    
	rawdigis.clear();
    rawdigis = theSiDigitalConverter->convertRaw(detAmpl, gainHandle, detID);
    push_link_raw(rawdigis, theLink, theCounterLink, detAmpl,detID);
    outrawdigi.data = rawdigis;
	
	//}
  }
}

void SiStripDigitizerAlgorithm::push_link(const DigitalVecType &digis,
					  const HitToDigisMapType& htd,
					  const HitCounterToDigisMapType& hctd,
					  const std::vector<double>& afterNoise,
					  unsigned int detID) {
  link_coll.clear();  
  for ( DigitalVecType::const_iterator i=digis.begin(); i!=digis.end(); i++) {
    // Instead of checking the validity of the links against the digis,
    //  let's loop over digis and push the corresponding link
    HitToDigisMapType::const_iterator mi(htd.find(i->strip()));  
    if (mi == htd.end()) continue;
    HitCounterToDigisMapType::const_iterator cmi(hctd.find(i->strip()));  
    std::map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
    for (std::vector < std::pair < const PSimHit*, Amplitude > >::const_iterator simul = 
	   (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
      totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
    }
    
    //--- include the noise as well
    double totalAmplitude1 = afterNoise[(*mi).first];
    
    //--- digisimlink
    int sim_counter=0; 
    for (std::map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	 iter != totalAmplitudePerSimHit.end(); iter++){
      float threshold = 0.;
      float fraction = (*iter).second/totalAmplitude1;
      if ( fraction >= threshold) {
	// Noise fluctuation could make fraction>1. Unphysical, set it by hand = 1.
	if(fraction > 1.) fraction = 1.;
	for (std::vector < std::pair < const PSimHit*, int > >::const_iterator 
	       simcount = (*cmi).second.begin() ; simcount != (*cmi).second.end(); simcount ++){
	  if((*iter).first == (*simcount).first) sim_counter = (*simcount).second;
	}
	link_coll.push_back(StripDigiSimLink( (*mi).first, //channel
					      ((*iter).first)->trackId(), //simhit trackId
					      sim_counter, //simhit counter
					      ((*iter).first)->eventId(), //simhit eventId
					      fraction)); //fraction
      }
    }
  }
}

void SiStripDigitizerAlgorithm::push_link_raw(const DigitalRawVecType &digis,
					      const HitToDigisMapType& htd,
					      const HitCounterToDigisMapType& hctd,
					      const std::vector<double>& afterNoise,
					      unsigned int detID) {
  link_coll.clear();  
  int nstrip = -1;
  for ( DigitalRawVecType::const_iterator i=digis.begin(); i!=digis.end(); i++) {
    nstrip++;
    // Instead of checking the validity of the links against the digis,
    //  let's loop over digis and push the corresponding link
    HitToDigisMapType::const_iterator mi(htd.find(nstrip));  
    HitCounterToDigisMapType::const_iterator cmi(hctd.find(nstrip));  
    if (mi == htd.end()) continue;
    std::map<const PSimHit *, Amplitude> totalAmplitudePerSimHit;
    for (std::vector < std::pair < const PSimHit*, Amplitude > >::const_iterator simul = 
	   (*mi).second.begin() ; simul != (*mi).second.end(); simul ++){
      totalAmplitudePerSimHit[(*simul).first] += (*simul).second;
    }
    
    //--- include the noise as well
    double totalAmplitude1 = afterNoise[(*mi).first];
    
    //--- digisimlink
    int sim_counter_raw=0;
    for (std::map<const PSimHit *, Amplitude>::const_iterator iter = totalAmplitudePerSimHit.begin(); 
	 iter != totalAmplitudePerSimHit.end(); iter++){
      float threshold = 0.;
      float fraction = (*iter).second/totalAmplitude1;
      if (fraction >= threshold) {
	//Noise fluctuation could make fraction>1. Unphysical, set it by hand.
	if(fraction >1.) fraction = 1.;
	//add counter information
	for (std::vector < std::pair < const PSimHit*, int > >::const_iterator 
	       simcount = (*cmi).second.begin() ; simcount != (*cmi).second.end(); simcount ++){
	  if((*iter).first == (*simcount).first) sim_counter_raw = (*simcount).second;
	}
	link_coll.push_back(StripDigiSimLink( (*mi).first, //channel
					      ((*iter).first)->trackId(), //simhit trackId
					      sim_counter_raw, //simhit counter
					      ((*iter).first)->eventId(), //simhit eventId
					      fraction)); //fraction
      }
    }
  }
}
