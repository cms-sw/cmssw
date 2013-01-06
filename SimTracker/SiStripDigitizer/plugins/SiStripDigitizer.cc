// File: SiStripDigitizerAlgorithm.cc
// Description:  Class for digitization.

// system include files
#include <memory>

#include "SimTracker/Common/interface/SimHitSelectorFromDB.h"

#include "SiStripDigitizer.h"
#include "SiStripDigitizerAlgorithm.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

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

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

//Data Base infromations
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

SiStripDigitizer::SiStripDigitizer(const edm::ParameterSet& conf, edm::EDProducer& mixMod) : 
  gainLabel(conf.getParameter<std::string>("Gain")),
  hitsProducer(conf.getParameter<std::string>("hitsProducer")),
  trackerContainers(conf.getParameter<std::vector<std::string> >("ROUList")),
  ZSDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("ZSDigi")),
  SCDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("SCDigi")),
  VRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("VRDigi")),
  PRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("PRDigi")),
  geometryType(conf.getParameter<std::string>("GeometryType")),
  useConfFromDB(conf.getParameter<bool>("TrackerConfigurationFromDB")),
  zeroSuppression(conf.getParameter<bool>("ZeroSuppression"))
{ 
  const std::string alias("simSiStripDigis");
  
  mixMod.produces<edm::DetSetVector<SiStripDigi> >(ZSDigi).setBranchAlias(ZSDigi);
  mixMod.produces<edm::DetSetVector<SiStripRawDigi> >(SCDigi).setBranchAlias(alias + SCDigi);
  mixMod.produces<edm::DetSetVector<SiStripRawDigi> >(VRDigi).setBranchAlias(alias + VRDigi);
  mixMod.produces<edm::DetSetVector<SiStripRawDigi> >(PRDigi).setBranchAlias(alias + PRDigi);
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "SiStripDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  
  rndEngine = &(rng->getEngine());
  theDigiAlgo.reset(new SiStripDigitizerAlgorithm(conf,(*rndEngine)));

}

// Virtual destructor needed.
SiStripDigitizer::~SiStripDigitizer() { 
}  

void SiStripDigitizer::accumulateStripHits(edm::Handle<std::vector<PSimHit> > hSimHits,
					   const TrackerTopology *tTopo) {
  if(hSimHits.isValid()) {
    std::set<unsigned int> detIds;
    std::vector<PSimHit> const& simHits = *hSimHits.product();
    for(std::vector<PSimHit>::const_iterator it = simHits.begin(), itEnd = simHits.end(); it != itEnd; ++it) {
      unsigned int detId = (*it).detUnitId();
      if(detIds.insert(detId).second) {
        // The insert succeeded, so this detector element has not yet been processed.
        StripGeomDetUnit* stripdet = detectorUnits[detId];
        //access to magnetic field in global coordinates
        GlobalVector bfield = pSetup->inTesla(stripdet->surface().position());
        LogDebug ("Digitizer ") << "B-field(T) at " << stripdet->surface().position() << "(cm): "
                                << pSetup->inTesla(stripdet->surface().position());
        theDigiAlgo->accumulateSimHits(it, itEnd, stripdet, bfield, tTopo);
      }
    }
  }
}

// Functions that gets called by framework every event
  void
  SiStripDigitizer::accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHand;
    iSetup.get<IdealGeometryRecord>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    // Step A: Get Inputs
    for(vstring::const_iterator i = trackerContainers.begin(), iEnd = trackerContainers.end(); i != iEnd; ++i) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer, *i);

      iEvent.getByLabel(tag, simHits);
      accumulateStripHits(simHits,tTopo);
    }
  }

  void
  SiStripDigitizer::accumulate(PileUpEventPrincipal const& iEvent, edm::EventSetup const& iSetup) {

    edm::ESHandle<TrackerTopology> tTopoHand;
    iSetup.get<IdealGeometryRecord>().get(tTopoHand);
    const TrackerTopology *tTopo=tTopoHand.product();

    // Step A: Get Inputs
    for(vstring::const_iterator i = trackerContainers.begin(), iEnd = trackerContainers.end(); i != iEnd; ++i) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer, *i);

      iEvent.getByLabel(tag, simHits);
      accumulateStripHits(simHits,tTopo);
    }
  }


void SiStripDigitizer::initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
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
    DetId idet=DetId(detId);
    unsigned int isub=idet.subdetId();
    if((isub == StripSubdetector::TIB) ||
       (isub == StripSubdetector::TID) ||
       (isub == StripSubdetector::TOB) ||
       (isub == StripSubdetector::TEC)) {
      StripGeomDetUnit* stripdet = dynamic_cast<StripGeomDetUnit*>((*iu));
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

  
  // Step B: LOOP on StripGeomDetUnit
  theDigiVector.reserve(10000);
  theDigiVector.clear();

  for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
    if(useConfFromDB){
      //apply the cable map _before_ digitization: consider only the detis that are connected 
      if(theDetIdList.find((*iu)->geographicalId().rawId())==theDetIdList.end())
        continue;
    }
    StripGeomDetUnit* sgd = dynamic_cast<StripGeomDetUnit*>((*iu));
    if (sgd != 0){
      edm::DetSet<SiStripDigi> collectorZS((*iu)->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorRaw((*iu)->geographicalId().rawId());
      theDigiAlgo->digitize(collectorZS,collectorRaw,sgd,
	 	       gainHandle,thresholdHandle,noiseHandle,pedestalHandle);
      if(zeroSuppression){
        if(collectorZS.data.size()>0){
          theDigiVector.push_back(collectorZS);
        }
      }else{
        if(collectorRaw.data.size()>0){
          theRawDigiVector.push_back(collectorRaw);
        }
      }
    }
  }

  if(zeroSuppression){
    // Step C: create output collection
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_virginraw(new edm::DetSetVector<SiStripRawDigi>());
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_scopemode(new edm::DetSetVector<SiStripRawDigi>());
    std::auto_ptr<edm::DetSetVector<SiStripRawDigi> > output_processedraw(new edm::DetSetVector<SiStripRawDigi>());
    std::auto_ptr<edm::DetSetVector<SiStripDigi> > output(new edm::DetSetVector<SiStripDigi>(theDigiVector) );
    // Step D: write output to file
    iEvent.put(output, ZSDigi);
    iEvent.put(output_scopemode, SCDigi);
    iEvent.put(output_virginraw, VRDigi);
    iEvent.put(output_processedraw, PRDigi);
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
  }
}
