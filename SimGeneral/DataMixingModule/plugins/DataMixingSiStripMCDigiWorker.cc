// File: DataMixingSiStripMCDigiWorker.cc
// Description:  see DataMixingSiStripMCDigiWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
//
#include "CLHEP/Random/RandFlat.h"
//
#include "DataMixingSiStripMCDigiWorker.h"

using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingSiStripMCDigiWorker::DataMixingSiStripMCDigiWorker() { }

  // Constructor 
  DataMixingSiStripMCDigiWorker::DataMixingSiStripMCDigiWorker(const edm::ParameterSet& ps, 
							       edm::ConsumesCollector && iC) : 
    label_(ps.getParameter<std::string>("Label")),
    gainLabel(ps.getParameter<std::string>("Gain")),
    SingleStripNoise(ps.getParameter<bool>("SingleStripNoise")),
    peakMode(ps.getParameter<bool>("APVpeakmode")),
    theThreshold(ps.getParameter<double>("NoiseSigmaThreshold")),
    theElectronPerADC(ps.getParameter<double>( peakMode ? "electronPerAdcPeak" : "electronPerAdcDec" )),
    APVSaturationFromHIP_(ps.getParameter<bool>("APVSaturationFromHIP")),
    theFedAlgo(ps.getParameter<int>("FedAlgorithm_PM")),
    geometryType(ps.getParameter<std::string>("GeometryType")),
    theSiZeroSuppress(new SiStripFedZeroSuppression(theFedAlgo)),
    theSiDigitalConverter(new SiTrivialDigitalConverter(theElectronPerADC, false)) // no premixing

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    SistripLabelSig_   = ps.getParameter<edm::InputTag>("SistripLabelSig");
    SiStripPileInputTag_ = ps.getParameter<edm::InputTag>("SiStripPileInputTag");

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");
    SistripAPVListDM_= ps.getParameter<std::string>("SiStripAPVListDM");


    if(APVSaturationFromHIP_) { 
      SistripAPVLabelSig_ = ps.getParameter<edm::InputTag>("SistripAPVLabelSig");
      SiStripAPVPileInputTag_ = ps.getParameter<edm::InputTag>("SistripAPVPileInputTag");
      iC.consumes< std::vector<std::pair<int,std::bitset<6>> > >(SistripAPVLabelSig_);
    }
    iC.consumes<edm::DetSetVector<SiStripDigi>>(SistripLabelSig_);
    // clear local storage for this event                                                                     
    SiHitStorage_.clear();

    edm::Service<edm::RandomNumberGenerator> rng;
    if ( ! rng.isAvailable()) {
    throw cms::Exception("Psiguration")
      << "SiStripDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the psiguration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
    }
  
    theSiNoiseAdder.reset(new SiGaussianTailNoiseAdder(theThreshold));

    //    theSiZeroSuppress = new SiStripFedZeroSuppression(theFedAlgo);
    //theSiDigitalConverter(new SiTrivialDigitalConverter(theElectronPerADC));

  }
	       

  // Virtual destructor needed.
  DataMixingSiStripMCDigiWorker::~DataMixingSiStripMCDigiWorker() { 
  }  

  void DataMixingSiStripMCDigiWorker::initializeEvent(const edm::Event &e, edm::EventSetup const& iSetup) {
    // initialize individual detectors so we can copy real digitization code:

    iSetup.get<TrackerDigiGeometryRecord>().get(geometryType,pDD);

    for(auto iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); ++iu) {
      unsigned int detId = (*iu)->geographicalId().rawId();
      DetId idet=DetId(detId);
      unsigned int isub=idet.subdetId();
      if((isub == StripSubdetector::TIB) ||
	 (isub == StripSubdetector::TID) ||
	 (isub == StripSubdetector::TOB) ||
	 (isub == StripSubdetector::TEC)) {

	auto stripdet = dynamic_cast<StripGeomDetUnit const*>((*iu));
	assert(stripdet != 0);
	DMinitializeDetUnit(stripdet, iSetup);
      }
    }
  }


  void DataMixingSiStripMCDigiWorker::DMinitializeDetUnit(StripGeomDetUnit const * det, const edm::EventSetup& iSetup ) { 

    edm::ESHandle<SiStripBadStrip> deadChannelHandle;
    iSetup.get<SiStripBadChannelRcd>().get(deadChannelHandle);

    unsigned int detId = det->geographicalId().rawId();
    int numStrips = (det->specificTopology()).nstrips();  

    SiStripBadStrip::Range detBadStripRange = deadChannelHandle->getRange(detId);
    //storing the bad strip of the the module. the module is not removed but just signal put to 0
    std::vector<bool>& badChannels = allBadChannels[detId];
    std::vector<bool>& hipChannels = allHIPChannels[detId];
    badChannels.clear();
    badChannels.insert(badChannels.begin(), numStrips, false);
    hipChannels.clear();
    hipChannels.insert(hipChannels.begin(), numStrips, false);

    for(SiStripBadStrip::ContainerIterator it = detBadStripRange.first; it != detBadStripRange.second; ++it) {
      SiStripBadStrip::data fs = deadChannelHandle->decode(*it);
      for(int strip = fs.firstStrip; strip < fs.firstStrip + fs.range; ++strip) badChannels[strip] = true;
    }
    firstChannelsWithSignal[detId] = numStrips;
    lastChannelsWithSignal[detId]= 0;
  }


  void DataMixingSiStripMCDigiWorker::addSiStripSignals(const edm::Event &e) { 
    // fill in maps of hits

    Handle< edm::DetSetVector<SiStripDigi> >  input;

    if( e.getByLabel(SistripLabelSig_,input) ) {
      OneDetectorMap LocalMap;

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<SiStripDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiStripMCDigiWorker")  << "Processing DetID " << DSViter->id;
#endif

	LocalMap.clear();
	LocalMap.reserve((DSViter->data).size());
	LocalMap.insert(LocalMap.end(),(DSViter->data).begin(),(DSViter->data).end());	
	
	SiHitStorage_.insert( SiGlobalIndex::value_type( DSViter->id, LocalMap ) );
      }
 
    }

    // keep here for future reference.  In current implementation, HIP killing is done once in PU file
    /*  if(APVSaturationFromHIP_) {
      Handle<std::vector<std::pair<int,std::bitset<6>> > >  APVinput;

      if( e.getByLabel(SistripAPVLabelSig_,APVinput) ) {

	std::vector<std::pair<int,std::bitset<6>> >::const_iterator entry = APVinput->begin();
	for( ; entry != APVinput->end(); entry++) {
	  theAffectedAPVmap_.insert(APVMap::value_type(entry->first, entry->second));
	}
      }
      } */

  } // end of addSiStripSignals



  void DataMixingSiStripMCDigiWorker::addSiStripPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                                  ModuleCallingContext const* mcc) {
    LogDebug("DataMixingSiStripMCDigiWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    std::shared_ptr<Wrapper<edm::DetSetVector<SiStripDigi> >  const> inputPTR =
      getProductByTag<edm::DetSetVector<SiStripDigi> >(*ep, SiStripPileInputTag_, mcc);

    if(inputPTR ) {

      const edm::DetSetVector<SiStripDigi>  *input = const_cast< edm::DetSetVector<SiStripDigi> * >(inputPTR->product());

      // Handle< edm::DetSetVector<SiStripDigi> >  input;

      // if( e->getByLabel(Sistripdigi_collectionPile_.label(),SistripLabelPile_.label(),input) ) {

      OneDetectorMap LocalMap;

      //loop on all detsets (detectorIDs) inside the input collection
      edm::DetSetVector<SiStripDigi>::const_iterator DSViter=input->begin();
      for (; DSViter!=input->end();DSViter++){

#ifdef DEBUG
	LogDebug("DataMixingSiStripMCDigiWorker")  << "Pileups: Processing DetID " << DSViter->id;
#endif

	// find correct local map (or new one) for this detector ID

	SiGlobalIndex::const_iterator itest;

	itest = SiHitStorage_.find(DSViter->id);

	if(itest!=SiHitStorage_.end()) {  // this detID already has hits, add to existing map

	  LocalMap = itest->second;

	  // fill in local map with extra channels
	  LocalMap.insert(LocalMap.end(),(DSViter->data).begin(),(DSViter->data).end());
	  std::stable_sort(LocalMap.begin(),LocalMap.end(),DataMixingSiStripMCDigiWorker::StrictWeakOrdering());
	  SiHitStorage_[DSViter->id]=LocalMap;
	  
	}
	else{ // fill local storage with this information, put in global collection

	  LocalMap.clear();
	  LocalMap.reserve((DSViter->data).size());
	  LocalMap.insert(LocalMap.end(),(DSViter->data).begin(),(DSViter->data).end());

	  SiHitStorage_.insert( SiGlobalIndex::value_type( DSViter->id, LocalMap ) );
	}
      }

      if(APVSaturationFromHIP_) {
	std::shared_ptr<Wrapper<std::vector<std::pair<int,std::bitset<6>> >  >  const> inputAPVPTR =
	  getProductByTag< std::vector<std::pair<int,std::bitset<6>> > >(*ep, SiStripAPVPileInputTag_, mcc);

	if(inputAPVPTR) {

	  const std::vector<std::pair<int,std::bitset<6>> >   *APVinput = const_cast< std::vector<std::pair<int,std::bitset<6>> >  * >(inputAPVPTR->product());

	  std::vector<std::pair<int,std::bitset<6>> >::const_iterator entry = APVinput->begin();
	  for( ; entry != APVinput->end(); entry++) {
	    theAffectedAPVmap_.insert( APVMap::value_type(entry->first, entry->second ));
	  }
	}
      }
    }
  }


 
  void DataMixingSiStripMCDigiWorker::putSiStrip(edm::Event &e, edm::EventSetup const& iSetup) {

    // set up machinery to do proper noise adding:
    edm::ESHandle<SiStripGain> gainHandle;
    edm::ESHandle<SiStripNoises> noiseHandle;
    edm::ESHandle<SiStripThreshold> thresholdHandle;
    edm::ESHandle<SiStripPedestals> pedestalHandle;
    edm::ESHandle<SiStripBadStrip> deadChannelHandle;
    iSetup.get<SiStripGainSimRcd>().get(gainLabel,gainHandle);
    iSetup.get<SiStripNoisesRcd>().get(noiseHandle);
    iSetup.get<SiStripThresholdRcd>().get(thresholdHandle);
    iSetup.get<SiStripPedestalsRcd>().get(pedestalHandle);

    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

    std::map< int,std::bitset<6>>  DeadAPVList;
    DeadAPVList.clear();


    // First, have to convert all ADC counts to raw pulse heights so that values can be added properly
    // In PreMixing, pulse heights are saved with ADC = sqrt(9.0*PulseHeight) - have to undo. 

    // This is done here because it's the only place we have access to EventSetup 
    // Simultaneously, merge lists of hit channels in each DetId.                
    // Signal Digis are in the list first, have to merge lists of hit strips on the fly,
    // add signals on duplicates later                                                       

    OneDetectorRawMap LocalRawMap;

    // Now, loop over hits and add them to the map in the proper sorted order        
    // Note: We are assuming that the hits from the Signal events have been created in
    // "PreMix" mode, rather than in the standard ADC conversion routines.  If not, this
    // doesn't work at all.

    // At the moment, both Signal and Reconstituted PU hits have the same compression algorithm.
    // If this were different, and one needed gains, the conversion back to pulse height can only
    // be done in this routine.  So, yes, there is an extra loop over hits here in the current code,
    // because, in principle, one could convert to pulse height during the read/store phase.    

    for(SiGlobalIndex::const_iterator IDet = SiHitStorage_.begin();
        IDet != SiHitStorage_.end(); IDet++) {

      uint32_t detID = IDet->first;

      OneDetectorMap LocalMap = IDet->second;

      //loop over hit strips for this DetId, do conversion to pulse height, store.

      LocalRawMap.clear();

      OneDetectorMap::const_iterator iLocal  = LocalMap.begin();
      for(;iLocal != LocalMap.end(); ++iLocal) {

        uint16_t currentStrip = iLocal->strip();
        float signal = float(iLocal->adc());
        if(iLocal->adc() == 1022) signal = 1500.;  // average values for overflows
        if(iLocal->adc() == 1023) signal = 3000.;

        //convert signals back to raw counts 

        float ReSignal = signal*signal/9.0;  // The PreMixing conversion is adc = sqrt(9.0*pulseHeight)

        RawDigi NewRawDigi = std::make_pair(currentStrip,ReSignal);

        LocalRawMap.push_back(NewRawDigi);

      }

      // save information for this detiD into global map
      SiRawDigis_.insert( SiGlobalRawIndex::value_type( detID, LocalRawMap ) );
    }

    // If we are killing APVs, merge list of dead ones before we digitize

    int NumberOfBxBetweenHIPandEvent=1e3;

    if(APVSaturationFromHIP_) { 

      // calculate affected BX parameter

      bool HasAtleastOneAffectedAPV=false;
      while(!HasAtleastOneAffectedAPV){
	for(int bx=floor(300.0/25.0);bx>0;bx--){ //Reminder: make these numbers not hard coded!!
	  float temp=CLHEP::RandFlat::shoot(engine)<0.5?1:0;
	  if(temp==1 && bx<NumberOfBxBetweenHIPandEvent){
	    NumberOfBxBetweenHIPandEvent=bx;
	    HasAtleastOneAffectedAPV=true;
	  }
	}
      }

      APVMap::const_iterator iAPVchk;
      uint32_t formerID = 0;
      uint32_t currentID;
      std::bitset<6> NewAPVBits;

      for(APVMap::const_iterator iAPV  = theAffectedAPVmap_.begin();
	  iAPV != theAffectedAPVmap_.end(); ++iAPV) {

	currentID = iAPV->first;

	if (currentID == formerID) { // we have to OR these
	  for( int ibit=0; ibit<6; ++ibit){
	    NewAPVBits[ibit] = NewAPVBits[ibit]||(iAPV->second)[ibit];
	  }
	}
	else {
	  DeadAPVList[currentID]=NewAPVBits;          
	  //save pointers for next iteration                                 
	  formerID = currentID;
	  NewAPVBits = iAPV->second;
	}

	iAPVchk = iAPV;
	if((++iAPVchk) == theAffectedAPVmap_.end()) {  //make sure not to lose the last one 
	  DeadAPVList[currentID]=NewAPVBits;          
	}
      }

    }
    // 

    //  Ok, done with merging raw signals and APVs - now add signals on duplicate strips 

    // collection of Digis to put in the event
    std::vector< edm::DetSet<SiStripDigi> > vSiStripDigi;

    // loop through our collection of detectors, merging hits and making a new list of "signal" digis

    // clear some temporary storage for later digitization:

    signals_.clear();

    // big loop over Detector IDs:
    for(SiGlobalRawIndex::const_iterator IDet = SiRawDigis_.begin();
        IDet != SiRawDigis_.end(); IDet++) {

      uint32_t detID = IDet->first;

      SignalMapType Signals;
      Signals.clear();

      OneDetectorRawMap LocalMap = IDet->second;

      //counter variables
      int formerStrip = -1;
      int currentStrip;
      float ADCSum = 0;

      //loop over hit strips for this DetId, add duplicates 

      OneDetectorRawMap::const_iterator iLocalchk;
      OneDetectorRawMap::const_iterator iLocal  = LocalMap.begin();
      for(;iLocal != LocalMap.end(); ++iLocal) {

        currentStrip = iLocal->first;  // strip is first element 

        if (currentStrip == formerStrip) { // we have to add these digis together 

          ADCSum+=iLocal->second ;          // raw pulse height is second element. 
        }
        else{
          if(formerStrip!=-1){
            Signals.insert( std::make_pair(formerStrip, ADCSum));
          }
          // save pointers for next iteration 
          formerStrip = currentStrip;
          ADCSum = iLocal->second; // lone ADC  
        }

        iLocalchk = iLocal;
	if((++iLocalchk) == LocalMap.end()) {  //make sure not to lose the last one  
          Signals.insert( std::make_pair(formerStrip, ADCSum));
        }
      }
      // save merged map: 
      signals_.insert( std::make_pair( detID, Signals));
    }

    //Now, do noise, zero suppression, take into account bad channels, etc.
    // This section stolen from SiStripDigitizerAlgorithm
    // must loop over all detIds in the tracker to get all of the noise added properly.
    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){

      const StripGeomDetUnit* sgd = dynamic_cast<const StripGeomDetUnit*>((*iu));
      if (sgd != 0){

	uint32_t detID = sgd->geographicalId().rawId();

	edm::DetSet<SiStripDigi> SSD(detID); // Make empty collection with this detector ID

	int numStrips = (sgd->specificTopology()).nstrips();

	// see if there is some signal on this detector

	const SignalMapType* theSignal(getSignal(detID));

	std::vector<float> detAmpl(numStrips, 0.);
	if(theSignal) {
	  for(const auto& amp : *theSignal) {
	    detAmpl[amp.first] = amp.second;
	  }
	}

	//removing signal from the dead (and HIP effected) strips
	std::vector<bool>& badChannels = allBadChannels[detID];

        for(int strip =0; strip < numStrips; ++strip) {
          if(badChannels[strip]) detAmpl[strip] = 0.;
	}
         
	if(APVSaturationFromHIP_) {
	  std::bitset<6> & bs=DeadAPVList[detID];

	  if(bs.any()){
	    // Here below is the scaling function which describes the evolution of the baseline (i.e. how the charge is suppressed).
	    // This must be replaced as soon as we have a proper modeling of the baseline evolution from VR runs
	    float Shift=1-NumberOfBxBetweenHIPandEvent/floor(300.0/25.0); //Reminder: make these numbers not hardcoded!! 
	    float randomX=CLHEP::RandFlat::shoot(engine);
	    float scalingValue=(randomX-Shift)*10.0/7.0-3.0/7.0;

	    for(int strip =0; strip < numStrips; ++strip) {
	      if(!badChannels[strip] &&  bs[strip/128]==1){
		detAmpl[strip] *=scalingValue>0?scalingValue:0.0;
	      }
	    }
	  }
	}

	SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
	SiStripApvGain::Range detGainRange = gainHandle->getRange(detID);

	// Gain conversion is already done during signal adding
	//convert our signals back to raw counts so that we can add noise properly:

	/*
        if(theSignal) {
          for(unsigned int iv = 0; iv!=detAmpl.size(); iv++) {
	    float signal = detAmpl[iv];
	    if(signal > 0) {
	      float gainValue = gainHandle->getStripGain(iv, detGainRange);
	      signal *= theElectronPerADC/gainValue;
	      detAmpl[iv] = signal;
	    }
          }
        }
	*/
	
	//SiStripPedestals::Range detPedestalRange = pedestalHandle->getRange(detID);

	// -----------------------------------------------------------

        size_t firstChannelWithSignal = 0;
        size_t lastChannelWithSignal = numStrips;

        if(SingleStripNoise){
          //      std::cout<<"In SSN, detId="<<detID<<std::endl;                                                                                                   
	  std::vector<float> noiseRMSv;
          noiseRMSv.clear();
          noiseRMSv.insert(noiseRMSv.begin(),numStrips,0.);
          for(int strip=0; strip< numStrips; ++strip){
            if(!badChannels[strip]){
              float gainValue = gainHandle->getStripGain(strip, detGainRange);
              noiseRMSv[strip] = (noiseHandle->getNoise(strip,detNoiseRange))* theElectronPerADC/gainValue;
              //std::cout<<"<SiStripDigitizerAlgorithm::digitize>: gainValue: "<<gainValue<<"\tnoiseRMSv["<<strip<<"]: "<<noiseRMSv[strip]<<std::endl;             
            }
          }
          theSiNoiseAdder->addNoiseVR(detAmpl, noiseRMSv, engine);
        } else {
          int RefStrip = int(numStrips/2.);
          while(RefStrip<numStrips&&badChannels[RefStrip]){ //if the refstrip is bad, I move up to when I don't find it                                            
            RefStrip++;
          }
          if(RefStrip<numStrips){
            float RefgainValue = gainHandle->getStripGain(RefStrip, detGainRange);
            float RefnoiseRMS = noiseHandle->getNoise(RefStrip,detNoiseRange) *theElectronPerADC/RefgainValue;

            theSiNoiseAdder->addNoise(detAmpl,firstChannelWithSignal,lastChannelWithSignal,numStrips,RefnoiseRMS, engine);
            //std::cout<<"<SiStripDigitizerAlgorithm::digitize>: RefgainValue: "<<RefgainValue<<"\tRefnoiseRMS: "<<RefnoiseRMS<<std::endl;                         
          }
        }
	
	DigitalVecType digis;
	theSiZeroSuppress->suppress(theSiDigitalConverter->convert(detAmpl, gainHandle, detID), digis, detID,noiseHandle,thresholdHandle);


	SSD.data = digis;
	//	if(digis.size() > 0) {
	//  std::cout << " Real SiS Mixed Digi: " << detID << " ADC values ";
	//  for(const auto& iDigi : digis) { std::cout << iDigi.adc() << " " ;}
	//  std::cout << std::endl;
	//}

	// stick this into the global vector of detector info
	vSiStripDigi.push_back(SSD);
	
      } // end of loop over one detector

    } // end of big loop over all detector IDs

    // put the collection of digis in the event   
    LogInfo("DataMixingSiStripMCDigiWorker") << "total # Merged strips: " << vSiStripDigi.size() ;

    // make new digi collection
    
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > MySiStripDigis(new edm::DetSetVector<SiStripDigi>(vSiStripDigi) );

    // put collection

    e.put( MySiStripDigis, SiStripDigiCollectionDM_ );

    // clear local storage for this event
    SiHitStorage_.clear();
    SiRawDigis_.clear();
    signals_.clear();
  }

} //edm
