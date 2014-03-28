// File: DataMixingSiStripMCDigiWorker.cc
// Description:  see DataMixingSiStripMCDigiWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
//
//
#include "DataMixingSiStripMCDigiWorker.h"

using namespace std;

namespace edm
{

  // Virtual constructor

  DataMixingSiStripMCDigiWorker::DataMixingSiStripMCDigiWorker() { }

  // Constructor 
  DataMixingSiStripMCDigiWorker::DataMixingSiStripMCDigiWorker(const edm::ParameterSet& ps) : 
    label_(ps.getParameter<std::string>("Label")),
    gainLabel(ps.getParameter<std::string>("Gain")),
    peakMode(ps.getParameter<bool>("APVpeakmode")),
    theThreshold(ps.getParameter<double>("NoiseSigmaThreshold")),
    theElectronPerADC(ps.getParameter<double>( peakMode ? "electronPerAdcPeak" : "electronPerAdcDec" )),
    theFedAlgo(ps.getParameter<int>("FedAlgorithm")),
    geometryType(ps.getParameter<std::string>("GeometryType")),
    theSiZeroSuppress(new SiStripFedZeroSuppression(theFedAlgo)),
    theSiDigitalConverter(new SiTrivialDigitalConverter(theElectronPerADC))

  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    SistripLabelSig_   = ps.getParameter<edm::InputTag>("SistripLabelSig");
    SiStripPileInputTag_ = ps.getParameter<edm::InputTag>("SiStripPileInputTag");

    SiStripDigiCollectionDM_  = ps.getParameter<std::string>("SiStripDigiCollectionDM");

    // clear local storage for this event                                                                     
    SiHitStorage_.clear();

    edm::Service<edm::RandomNumberGenerator> rng;
    if ( ! rng.isAvailable()) {
    throw cms::Exception("Psiguration")
      << "SiStripDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the psiguration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
    }
  
    rndEngine = &(rng->getEngine());

    theSiNoiseAdder.reset(new SiGaussianTailNoiseAdder(theThreshold, (*rndEngine) ));
    //    theSiZeroSuppress = new SiStripFedZeroSuppression(theFedAlgo);
    //theSiDigitalConverter(new SiTrivialDigitalConverter(theElectronPerADC));

  }
	       

  // Virtual destructor needed.
  DataMixingSiStripMCDigiWorker::~DataMixingSiStripMCDigiWorker() { 
  }  

  void DataMixingSiStripMCDigiWorker::initializeEvent(const edm::Event &e, edm::EventSetup const& iSetup) {
    // initialize individual detectors so we can copy real digitization code:

    iSetup.get<TrackerDigiGeometryRecord>().get(geometryType,pDD);

    for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); ++iu) {
      unsigned int detId = (*iu)->geographicalId().rawId();
      DetId idet=DetId(detId);
      unsigned int isub=idet.subdetId();
      if((isub == StripSubdetector::TIB) ||
	 (isub == StripSubdetector::TID) ||
	 (isub == StripSubdetector::TOB) ||
	 (isub == StripSubdetector::TEC)) {
	StripGeomDetUnit* stripdet = dynamic_cast<StripGeomDetUnit*>((*iu));
	assert(stripdet != 0);
	DMinitializeDetUnit(stripdet, iSetup);
      }
    }


  }


  void DataMixingSiStripMCDigiWorker::DMinitializeDetUnit(StripGeomDetUnit* det, const edm::EventSetup& iSetup ) { 

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
  } // end of addSiStripSignals



  void DataMixingSiStripMCDigiWorker::addSiStripPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,
                                                  ModuleCallingContext const* mcc) {
    LogDebug("DataMixingSiStripMCDigiWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // fill in maps of hits; same code as addSignals, except now applied to the pileup events

    boost::shared_ptr<Wrapper<edm::DetSetVector<SiStripDigi> >  const> inputPTR =
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


    // collection of Digis to put in the event
    std::vector< edm::DetSet<SiStripDigi> > vSiStripDigi;

    // loop through our collection of detectors, merging hits and putting new ones in the output

    // big loop over Detector IDs:

    for(SiGlobalIndex::const_iterator IDet = SiHitStorage_.begin();
	IDet != SiHitStorage_.end(); IDet++) {

      unsigned int detID = IDet->first;

      edm::DetSet<SiStripDigi> SSD(detID); // Make empty collection with this detector ID

      // get the right detector element for this det ID:
      const GeomDetUnit* it = pDD->idToDetUnit(DetId(detID));
      const StripGeomDetUnit* det = dynamic_cast<const StripGeomDetUnit*>((it));
      int numStrips = (det->specificTopology()).nstrips();

      std::vector<float> detAmpl(numStrips, 0.);
	
      OneDetectorMap LocalMap = IDet->second;

      //counter variables
      int formerStrip = -1;
      int currentStrip;
      int ADCSum = 0;

      //loop over hit strips for this DetId, add duplicates

      OneDetectorMap::const_iterator iLocalchk;
      OneDetectorMap::const_iterator iLocal  = LocalMap.begin();
      for(;iLocal != LocalMap.end(); ++iLocal) {

	currentStrip = iLocal->strip(); 

	if (currentStrip == formerStrip) { // we have to add these digis together
	  ADCSum+=iLocal->adc();          // on every element...
	}
	else{
	  if(formerStrip!=-1){
	    detAmpl[formerStrip] = ADCSum;

	    //if (ADCSum > 511) ADCSum = 255;
	    //else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;
	    //SiStripDigi aHit(formerStrip, ADCSum);
	    //SSD.push_back( aHit );	  
	  }
	  // save pointers for next iteration
	  formerStrip = currentStrip;
	  ADCSum = iLocal->adc();
	}

	iLocalchk = iLocal;
	if((++iLocalchk) == LocalMap.end()) {  //make sure not to lose the last one

	  detAmpl[formerStrip] = ADCSum;

	  //	  if (ADCSum > 511) ADCSum = 255;
	  //else if (ADCSum > 253 && ADCSum < 512) ADCSum = 254;
	  //SSD.push_back( SiStripDigi(formerStrip, ADCSum) );	  
	} 

	//Now, do noise, zero suppression, take into account bad channels, etc.
	// This section stolen from SiStripDigitizerAlgorithm

	//removing signal from the dead (and HIP effected) strips
	std::vector<bool>& badChannels = allBadChannels[detID];
	for(int strip =0; strip < numStrips; ++strip) if(badChannels[strip]) detAmpl[strip] = 0.;

	SiStripNoises::Range detNoiseRange = noiseHandle->getRange(detID);
	SiStripApvGain::Range detGainRange = gainHandle->getRange(detID);
	//SiStripPedestals::Range detPedestalRange = pedestalHandle->getRange(detID);

	// -----------------------------------------------------------

	auto& firstChannelWithSignal = firstChannelsWithSignal[detID];
	auto& lastChannelWithSignal = lastChannelsWithSignal[detID];

	int RefStrip = int(numStrips/2.);
	while(RefStrip<numStrips&&badChannels[RefStrip]){ //if the refstrip is bad, I move up to when I don't find it
	  RefStrip++;
	}
	if(RefStrip<numStrips){
	  float noiseRMS = noiseHandle->getNoise(RefStrip,detNoiseRange);
	  float gainValue = gainHandle->getStripGain(RefStrip, detGainRange);
	  theSiNoiseAdder->addNoise(detAmpl,firstChannelWithSignal,lastChannelWithSignal,numStrips,noiseRMS*theElectronPerADC/gainValue);
	}
	
	DigitalVecType digis;
	theSiZeroSuppress->suppress(theSiDigitalConverter->convert(detAmpl, gainHandle, detID), digis, detID,noiseHandle,thresholdHandle);

	SSD.data = digis;
	
      } // end of loop over one detector

      // stick this into the global vector of detector info
      vSiStripDigi.push_back(SSD);

    } // end of big loop over all detector IDs

    // put the collection of digis in the event   
    LogInfo("DataMixingSiStripMCDigiWorker") << "total # Merged strips: " << vSiStripDigi.size() ;

    // make new digi collection
    
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > MySiStripDigis(new edm::DetSetVector<SiStripDigi>(vSiStripDigi) );

    // put collection

    e.put( MySiStripDigis, SiStripDigiCollectionDM_ );

    // clear local storage for this event
    SiHitStorage_.clear();
  }

} //edm
