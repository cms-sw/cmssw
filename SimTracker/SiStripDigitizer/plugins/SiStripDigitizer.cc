// File: SiStripDigitizerAlgorithm.cc
// Description:  Class for digitization.

// Modified 15/May/2013 mark.grimes@bristol.ac.uk - Modified so that the digi-sim link has the correct
// index for the sim hits stored. It was previously always set to zero (I won't mention that it was
// me who originally wrote that).

// system include files
#include <memory>

#include "SimTracker/Common/interface/SimHitSelectorFromDB.h"

#include "SiStripDigitizer.h"
#include "SiStripDigitizerAlgorithm.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
//needed for the magnetic field:
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//Data Base infromations
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

SiStripDigitizer::SiStripDigitizer(const edm::ParameterSet& conf, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC) : 
  gainLabel(conf.getParameter<std::string>("Gain")),
  hitsProducer(conf.getParameter<std::string>("hitsProducer")),
  trackerContainers(conf.getParameter<std::vector<std::string> >("ROUList")),
  ZSDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("ZSDigi")),
  SCDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("SCDigi")),
  VRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("VRDigi")),
  PRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("PRDigi")),
  geometryType(conf.getParameter<std::string>("GeometryType")),
  useConfFromDB(conf.getParameter<bool>("TrackerConfigurationFromDB")),
  zeroSuppression(conf.getParameter<bool>("ZeroSuppression")),
  makeDigiSimLinks_(conf.getUntrackedParameter<bool>("makeDigiSimLinks", false))
{ 
  const std::string alias("simSiStripDigis");
  
  mixMod.produces<edm::DetSetVector<SiStripDigi> >(ZSDigi).setBranchAlias(ZSDigi);
  mixMod.produces<edm::DetSetVector<SiStripRawDigi> >(SCDigi).setBranchAlias(alias + SCDigi);
  mixMod.produces<edm::DetSetVector<SiStripRawDigi> >(VRDigi).setBranchAlias(alias + VRDigi);
  mixMod.produces<edm::DetSetVector<SiStripRawDigi> >(PRDigi).setBranchAlias(alias + PRDigi);
  mixMod.produces<edm::DetSetVector<StripDigiSimLink> >().setBranchAlias(alias + "siStripDigiSimLink");
  mixMod.produces<std::vector<std::pair<int,std::bitset<6>>>>("AffectedAPVList").setBranchAlias(alias + "AffectedAPV");
  for(auto const& trackerContainer : trackerContainers) {
    edm::InputTag tag(hitsProducer, trackerContainer);
    iC.consumes<std::vector<PSimHit> >(edm::InputTag(hitsProducer, trackerContainer));  
  }
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "SiStripDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  theDigiAlgo.reset(new SiStripDigitizerAlgorithm(conf));
}

// Virtual destructor needed.
SiStripDigitizer::~SiStripDigitizer() { 
}  

void SiStripDigitizer::accumulateStripHits(edm::Handle<std::vector<PSimHit> > hSimHits,
					   const TrackerTopology *tTopo, size_t globalSimHitIndex, const unsigned int tofBin, CLHEP::HepRandomEngine* engine ) {
  // globalSimHitIndex is the index the sim hit will have when it is put in a collection
  // of sim hits for all crossings. This is only used when creating digi-sim links if
  // configured to do so.

  if(hSimHits.isValid()) {
    std::set<unsigned int> detIds;
    std::vector<PSimHit> const& simHits = *hSimHits.product();
    for(std::vector<PSimHit>::const_iterator it = simHits.begin(), itEnd = simHits.end(); it != itEnd; ++it, ++globalSimHitIndex ) {
      unsigned int detId = (*it).detUnitId();
      if(detIds.insert(detId).second) {
        // The insert succeeded, so this detector element has not yet been processed.
	assert(detectorUnits[detId]);
	if(detectorUnits[detId]->type().isTrackerStrip()) { // this test can be removed and replaced by stripdet!=0
	  auto stripdet = detectorUnits[detId];
	  //access to magnetic field in global coordinates
	  GlobalVector bfield = pSetup->inTesla(stripdet->surface().position());
	  LogDebug ("Digitizer ") << "B-field(T) at " << stripdet->surface().position() << "(cm): "
				  << pSetup->inTesla(stripdet->surface().position());
	  theDigiAlgo->accumulateSimHits(it, itEnd, globalSimHitIndex, tofBin, stripdet, bfield, tTopo, engine);
	}
      }
    } // end of loop over sim hits
  }
}

// Functions that gets called by framework every event
  void
  SiStripDigitizer::accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHand;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    // Step A: Get Inputs
    for(auto const& trackerContainer : trackerContainers) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer, trackerContainer);
      unsigned int tofBin = StripDigiSimLink::LowTof;
      if (trackerContainer.find(std::string("HighTof")) != std::string::npos) tofBin = StripDigiSimLink::HighTof;

      iEvent.getByLabel(tag, simHits);
      accumulateStripHits(simHits,tTopo,crossingSimHitIndexOffset_[tag.encode()], tofBin, randomEngine(iEvent.streamID()));
      // Now that the hits have been processed, I'll add the amount of hits in this crossing on to
      // the global counter. Next time accumulateStripHits() is called it will count the sim hits
      // as though they were on the end of this collection.
      // Note that this is only used for creating digi-sim links (if configured to do so).
      if( simHits.isValid() ) crossingSimHitIndexOffset_[tag.encode()]+=simHits->size();
    }
  }

  void
  SiStripDigitizer::accumulate(PileUpEventPrincipal const& iEvent, edm::EventSetup const& iSetup, edm::StreamID const& streamID) {

    edm::ESHandle<TrackerTopology> tTopoHand;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    //Re-compute luminosity for accumulation for HIP effects
    PileupInfo_ = getEventPileupInfo();
    theDigiAlgo->calculateInstlumiScale(PileupInfo_);

    // Step A: Get Inputs
    for(auto const& trackerContainer : trackerContainers) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer, trackerContainer);
      unsigned int tofBin = StripDigiSimLink::LowTof;
      if (trackerContainer.find(std::string("HighTof")) != std::string::npos) tofBin = StripDigiSimLink::HighTof; 

      iEvent.getByLabel(tag, simHits);
      accumulateStripHits(simHits,tTopo,crossingSimHitIndexOffset_[tag.encode()], tofBin, randomEngine(streamID));
      // Now that the hits have been processed, I'll add the amount of hits in this crossing on to
      // the global counter. Next time accumulateStripHits() is called it will count the sim hits
      // as though they were on the end of this collection.
      // Note that this is only used for creating digi-sim links (if configured to do so).
      if( simHits.isValid() ) crossingSimHitIndexOffset_[tag.encode()]+=simHits->size();
    }
  }


void SiStripDigitizer::initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // Make sure that the first crossing processed starts indexing the sim hits from zero.
  // This variable is used so that the sim hits from all crossing frames have sequential
  // indices used to create the digi-sim link (if configured to do so) rather than starting
  // from zero for each crossing.
  crossingSimHitIndexOffset_.clear();
  theAffectedAPVvector.clear();
  // Step A: Get Inputs

  if(useConfFromDB){
    edm::ESHandle<SiStripDetCabling> detCabling;
    iSetup.get<SiStripDetCablingRcd>().get(detCabling);
    detCabling->addConnected(theDetIdList);
  }

  theDigiAlgo->initializeEvent(iSetup);

  iSetup.get<TrackerDigiGeometryRecord>().get(geometryType,pDD);
  iSetup.get<IdealMagneticFieldRecord>().get(pSetup);

  // FIX THIS! We only need to clear and (re)fill detectorUnits when the geometry type IOV changes.  Use ESWatcher to determine this.
  bool changes = true;
  if(changes) { // Replace with ESWatcher
    detectorUnits.clear();
  }
  for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); ++iu) {
    unsigned int detId = (*iu)->geographicalId().rawId();
    if((*iu)->type().isTrackerStrip()) {
      auto stripdet = dynamic_cast<StripGeomDetUnit const*>((*iu));
      assert(stripdet != 0);
      if(changes) { // Replace with ESWatcher
        detectorUnits.insert(std::make_pair(detId, stripdet));
      }
      theDigiAlgo->initializeDetUnit(stripdet, iSetup);
    }
  }
}

void SiStripDigitizer::finalizeEvent(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripThreshold> thresholdHandle;
  edm::ESHandle<SiStripPedestals> pedestalHandle;
  iSetup.get<SiStripGainSimRcd>().get(gainLabel,gainHandle);
  iSetup.get<SiStripNoisesRcd>().get(noiseHandle);
  iSetup.get<SiStripThresholdRcd>().get(thresholdHandle);
  iSetup.get<SiStripPedestalsRcd>().get(pedestalHandle);

  std::vector<edm::DetSet<SiStripDigi> > theDigiVector;
  std::vector<edm::DetSet<SiStripRawDigi> > theRawDigiVector;
  std::unique_ptr< edm::DetSetVector<StripDigiSimLink> > pOutputDigiSimLink( new edm::DetSetVector<StripDigiSimLink> );

  // Step B: LOOP on StripGeomDetUnit
  theDigiVector.reserve(10000);
  theDigiVector.clear();

  for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
    if(useConfFromDB){
      //apply the cable map _before_ digitization: consider only the detis that are connected 
      if(theDetIdList.find((*iu)->geographicalId().rawId())==theDetIdList.end())
        continue;
    }
    auto sgd = dynamic_cast<StripGeomDetUnit const*>((*iu));
    if (sgd != 0){
      edm::DetSet<SiStripDigi> collectorZS((*iu)->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorRaw((*iu)->geographicalId().rawId());
      edm::DetSet<StripDigiSimLink> collectorLink((*iu)->geographicalId().rawId());
      theDigiAlgo->digitize(collectorZS,collectorRaw,collectorLink,sgd,
                            gainHandle,thresholdHandle,noiseHandle,pedestalHandle,theAffectedAPVvector,randomEngine(iEvent.streamID()));
      if(zeroSuppression){
        if(collectorZS.data.size()>0){
          theDigiVector.push_back(collectorZS);
          if( !collectorLink.data.empty() ) pOutputDigiSimLink->insert(collectorLink);
        }
      }else{
        if(collectorRaw.data.size()>0){
          theRawDigiVector.push_back(collectorRaw);
          if( !collectorLink.data.empty() ) pOutputDigiSimLink->insert(collectorLink);
        }
      }
    }
  }
  if(zeroSuppression){
    // Step C: create output collection
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > output_virginraw(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > output_scopemode(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi> > output_processedraw(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(theDigiVector) );
    std::unique_ptr<std::vector<std::pair<int,std::bitset<6>>> > AffectedAPVList(new std::vector<std::pair<int,std::bitset<6>>>(theAffectedAPVvector));

    // Step D: write output to file
    iEvent.put(std::move(output), ZSDigi);
    iEvent.put(std::move(output_scopemode), SCDigi);
    iEvent.put(std::move(output_virginraw), VRDigi);
    iEvent.put(std::move(output_processedraw), PRDigi);
    iEvent.put(std::move(AffectedAPVList),"AffectedAPVList");
    if( makeDigiSimLinks_ ) iEvent.put(std::move(pOutputDigiSimLink)); // The previous EDProducer didn't name this collection so I won't either
  }else{
    // Step C: create output collection
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_virginraw(new edm::DetSetVector<SiStripRawDigi>(theRawDigiVector));
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_scopemode(new edm::DetSetVector<SiStripRawDigi>());
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_processedraw(new edm::DetSetVector<SiStripRawDigi>());
    std::auto_ptr<edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>() );
    // Step D: write output to file
    iEvent.put(output, ZSDigi);
    iEvent.put(output_scopemode, SCDigi);
    iEvent.put(output_virginraw, VRDigi);
    iEvent.put(output_processedraw, PRDigi);
    if( makeDigiSimLinks_ ) iEvent.put( std::move(pOutputDigiSimLink) ); // The previous EDProducer didn't name this collection so I won't either
  }
}

CLHEP::HepRandomEngine* SiStripDigitizer::randomEngine(edm::StreamID const& streamID) {
  unsigned int index = streamID.value();
  if(index >= randomEngines_.size()) {
    randomEngines_.resize(index + 1, nullptr);
  }
  CLHEP::HepRandomEngine* ptr = randomEngines_[index];
  if(!ptr) {
    edm::Service<edm::RandomNumberGenerator> rng;
    ptr = &rng->getEngine(streamID);
    randomEngines_[index] = ptr;
  }
  return ptr;
}
