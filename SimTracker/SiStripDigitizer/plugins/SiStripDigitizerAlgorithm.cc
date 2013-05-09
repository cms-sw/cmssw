// File: SiStripDigitizerAlgorithm.cc
// Description:  Steering class for digitization.

#include <vector>
#include <algorithm>
#include <iostream>
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SiStripDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CLHEP/Random/RandFlat.h"

SiStripDigitizerAlgorithm::SiStripDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng):
  lorentzAngleName(conf.getParameter<std::string>("LorentzAngle")),
  theThreshold(conf.getParameter<double>("NoiseSigmaThreshold")),
  cmnRMStib(conf.getParameter<double>("cmnRMStib")),
  cmnRMStob(conf.getParameter<double>("cmnRMStob")),
  cmnRMStid(conf.getParameter<double>("cmnRMStid")),
  cmnRMStec(conf.getParameter<double>("cmnRMStec")),
  APVSaturationProb(conf.getParameter<double>("APVSaturationProb")),
  makeDigiSimLinks_(conf.getUntrackedParameter<bool>("makeDigiSimLinks", false)),
  peakMode(conf.getParameter<bool>("APVpeakmode")),
  noise(conf.getParameter<bool>("Noise")),
  RealPedestals(conf.getParameter<bool>("RealPedestals")), 
  SingleStripNoise(conf.getParameter<bool>("SingleStripNoise")),
  CommonModeNoise(conf.getParameter<bool>("CommonModeNoise")),
  BaselineShift(conf.getParameter<bool>("BaselineShift")),
  APVSaturationFromHIP(conf.getParameter<bool>("APVSaturationFromHIP")),
  theFedAlgo(conf.getParameter<int>("FedAlgorithm")),
  zeroSuppression(conf.getParameter<bool>("ZeroSuppression")),
  theElectronPerADC(conf.getParameter<double>( peakMode ? "electronPerAdcPeak" : "electronPerAdcDec" )),
  theTOFCutForPeak(conf.getParameter<double>("TOFCutForPeak")),
  theTOFCutForDeconvolution(conf.getParameter<double>("TOFCutForDeconvolution")),
  tofCut(peakMode ? theTOFCutForPeak : theTOFCutForDeconvolution),
  cosmicShift(conf.getUntrackedParameter<double>("CosmicDelayShift")),
  inefficiency(conf.getParameter<double>("Inefficiency")),
  pedOffset((unsigned int)conf.getParameter<double>("PedestalsOffset")),
  theSiHitDigitizer(new SiHitDigitizer(conf, eng)),
  theSiPileUpSignals(new SiPileUpSignals()),
  theSiNoiseAdder(new SiGaussianTailNoiseAdder(theThreshold, eng)),
  theSiDigitalConverter(new SiTrivialDigitalConverter(theElectronPerADC)),
  theSiZeroSuppress(new SiStripFedZeroSuppression(theFedAlgo)),
  theFlatDistribution(new CLHEP::RandFlat(eng, 0., 1.)) {

  if (peakMode) {
    LogDebug("StripDigiInfo")<<"APVs running in peak mode (poor time resolution)";
  } else {
    LogDebug("StripDigiInfo")<<"APVs running in deconvolution mode (good time resolution)";
  };
}

SiStripDigitizerAlgorithm::~SiStripDigitizerAlgorithm(){
}

void
SiStripDigitizerAlgorithm::initializeDetUnit(StripGeomDetUnit* det, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiStripBadStrip> deadChannelHandle;
  iSetup.get<SiStripBadChannelRcd>().get(deadChannelHandle);

  unsigned int detId = det->geographicalId().rawId();
  int numStrips = (det->specificTopology()).nstrips();  

  SiStripBadStrip::Range detBadStripRange = deadChannelHandle->getRange(detId);
  //storing the bad strip of the the module. the module is not removed but just signal put to 0
  std::vector<bool>& badChannels = allBadChannels[detId];
  badChannels.clear();
  badChannels.insert(badChannels.begin(), numStrips, false);
  for(SiStripBadStrip::ContainerIterator it = detBadStripRange.first; it != detBadStripRange.second; ++it) {
    SiStripBadStrip::data fs = deadChannelHandle->decode(*it);
    for(int strip = fs.firstStrip; strip < fs.firstStrip + fs.range; ++strip) badChannels[strip] = true;
  }
  firstChannelsWithSignal[detId] = numStrips;
  lastChannelsWithSignal[detId]= 0;
}

void
SiStripDigitizerAlgorithm::initializeEvent(const edm::EventSetup& iSetup) {
  theSiPileUpSignals->reset();

  //get gain noise pedestal lorentzAngle from ES handle
  edm::ESHandle<ParticleDataTable> pdt;
  iSetup.getData(pdt);
  setParticleDataTable(&*pdt);
  iSetup.get<SiStripLorentzAngleSimRcd>().get(lorentzAngleName,lorentzAngleHandle);
}

//  Run the algorithm for a given module
//  ------------------------------------

void
SiStripDigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
                                             std::vector<PSimHit>::const_iterator inputEnd,
                                             const StripGeomDetUnit* det,
                                             const GlobalVector& bfield,
					     const TrackerTopology *tTopo) {
  // produce SignalPoints for all SimHits in detector
  unsigned int detID = det->geographicalId().rawId();
  int numStrips = (det->specificTopology()).nstrips();  

  std::vector<bool>& badChannels = allBadChannels[detID];
  size_t thisFirstChannelWithSignal = numStrips;
  size_t thisLastChannelWithSignal = numStrips;

  float langle = (lorentzAngleHandle.isValid()) ? lorentzAngleHandle->getLorentzAngle(detID) : 0.;

  std::vector<double> locAmpl(numStrips, 0.);

  // Loop over hits

  uint32_t detId = det->geographicalId().rawId();
  // First: loop on the SimHits
  if(theFlatDistribution->fire()>inefficiency) {
    for (std::vector<PSimHit>::const_iterator simHitIter = inputBegin; simHitIter != inputEnd; ++simHitIter) {
      // skip hits not in this detector.
      if((*simHitIter).detUnitId() != detId) {
        continue;
      }
      // check TOF
      if (std::fabs(simHitIter->tof() - cosmicShift - det->surface().toGlobal(simHitIter->localPosition()).mag()/30.) < tofCut && simHitIter->energyLoss()>0) {
        size_t localFirstChannel = numStrips;
        size_t localLastChannel  = 0;
        // process the hit
        theSiHitDigitizer->processHit(&*simHitIter, *det, bfield, langle, locAmpl, localFirstChannel, localLastChannel, tTopo);
          
		  //APV Killer to simulate HIP effect
		  //------------------------------------------------------
		  
		  if(APVSaturationFromHIP&&!zeroSuppression){
		    int pdg_id = simHitIter->particleType();
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
				 		for(int strip = FirstAPV*128; strip < LastAPV*128 +128; ++strip) {
							badChannels[strip] = true;
						}
						//doing like that I remove the signal information only after the 
						//stip that got the HIP but it remains the signal of the previous
						//one. I'll make a further loop to remove all signal
			  		}
				}
			}
	      }             
		
    
        if(thisFirstChannelWithSignal > localFirstChannel) thisFirstChannelWithSignal = localFirstChannel;
        if(thisLastChannelWithSignal < localLastChannel) thisLastChannelWithSignal = localLastChannel;
      }
    } // end for
  }
  theSiPileUpSignals->add(detID, locAmpl, thisFirstChannelWithSignal, thisLastChannelWithSignal);

  if(firstChannelsWithSignal[detID] > thisFirstChannelWithSignal) firstChannelsWithSignal[detID] = thisFirstChannelWithSignal;
  if(lastChannelsWithSignal[detID] < thisLastChannelWithSignal) lastChannelsWithSignal[detID] = thisLastChannelWithSignal;
}

void
SiStripDigitizerAlgorithm::digitize(
			   edm::DetSet<SiStripDigi>& outdigi,
			   edm::DetSet<SiStripRawDigi>& outrawdigi,
			   const StripGeomDetUnit *det,
			   edm::ESHandle<SiStripGain> & gainHandle,
			   edm::ESHandle<SiStripThreshold> & thresholdHandle,
			   edm::ESHandle<SiStripNoises> & noiseHandle,
			   edm::ESHandle<SiStripPedestals> & pedestalHandle) {
  unsigned int detID = det->geographicalId().rawId();
  int numStrips = (det->specificTopology()).nstrips();  

  const SiPileUpSignals::SignalMapType* theSignal(theSiPileUpSignals->getSignal(detID));  

  std::vector<double> detAmpl(numStrips, 0.);
  if(theSignal) {
    for(const auto& amp : *theSignal) {
      detAmpl[amp.first] = amp.second;
    }
  }

  //removing signal from the dead (and HIP effected) strips
  std::vector<bool>& badChannels = allBadChannels[detID];
  for(int strip =0; strip < numStrips; ++strip) if(badChannels[strip]) detAmpl[strip] = 0.;

  SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
  SiStripApvGain::Range detGainRange = gainHandle->getRange(detID);
  SiStripPedestals::Range detPedestalRange = pedestalHandle->getRange(detID);

// -----------------------------------------------------------

  auto& firstChannelWithSignal = firstChannelsWithSignal[detID];
  auto& lastChannelWithSignal = lastChannelsWithSignal[detID];

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
    DigitalVecType digis;
    theSiZeroSuppress->suppress(theSiDigitalConverter->convert(detAmpl, gainHandle, detID), digis, detID,noiseHandle,thresholdHandle);
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
    
    DigitalRawVecType rawdigis = theSiDigitalConverter->convertRaw(detAmpl, gainHandle, detID);
    outrawdigi.data = rawdigis;
	
	//}
  }
}
