// File: DigiSimLinkAlgorithm.cc
// Description:  Class for digitization.

// system include files
#include <memory>

#include "DigiSimLinkProducer.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

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
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

//Data Base infromations
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"


//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

DigiSimLinkProducer::DigiSimLinkProducer(const edm::ParameterSet& conf) : 
  conf_(conf)
{
  alias = conf.getParameter<std::string>("@module_label");
  edm::ParameterSet ParamSet=conf_.getParameter<edm::ParameterSet>("DigiModeList");
  
  produces<edm::DetSetVector<StripDigiSimLink> >().setBranchAlias ( alias + "siStripDigiSimLink");
  trackerContainers.clear();
  trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");
  geometryType = conf.getParameter<std::string>("GeometryType");
  useConfFromDB = conf.getParameter<bool>("TrackerConfigurationFromDB");
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "DigiSimLinkProducer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  
  rndEngine       = &(rng->getEngine());
  zeroSuppression = conf_.getParameter<bool>("ZeroSuppression");
  theDigiAlgo = new DigiSimLinkAlgorithm(conf_,(*rndEngine));

}

// Virtual destructor needed.
DigiSimLinkProducer::~DigiSimLinkProducer() { 
  delete theDigiAlgo;
}  

// Functions that gets called by framework every event
void DigiSimLinkProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Step A: Get Inputs
  edm::ESHandle < ParticleDataTable > pdt;
  iSetup.getData( pdt );

  if(useConfFromDB){
    edm::ESHandle<SiStripDetCabling> detCabling;
    iSetup.get<SiStripDetCablingRcd>().get( detCabling );
    detCabling->addConnected(theDetIdList);
  }

  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  for(uint32_t i = 0; i< trackerContainers.size();i++){
    iEvent.getByLabel("mix",trackerContainers[i],cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }

  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
  
  //Loop on PSimHit
  SimHitMap.clear();
  
  //inside SimHitSelectorFromDb add the counter information from the original allhits collection 
  std::vector<std::pair<const PSimHit*,int> > trackerHits(SimHitSelectorFromDB_.getSimHit(allTrackerHits,theDetIdList));
  std::vector<std::pair<const PSimHit*,int> >::iterator isim;
  for (isim=trackerHits.begin() ; isim!= trackerHits.end();isim++) {
    //make a pair = <*isim, counter> and save also position in the vector for DigiSimLink
    SimHitMap[((*isim).first)->detUnitId()].push_back(*isim);
  }
  
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryType,pDD);
  
  edm::ESHandle<MagneticField> pSetup;
  iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
  
  //get gain noise pedestal lorentzAngle from ES handle
  edm::ESHandle<SiStripLorentzAngle> lorentzAngleHandle;
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripThreshold> thresholdHandle;
  edm::ESHandle<SiStripPedestals> pedestalHandle;
  edm::ESHandle<SiStripBadStrip> deadChannelHandle;
  std::string LAname = conf_.getParameter<std::string>("LorentzAngle");
  iSetup.get<SiStripLorentzAngleSimRcd>().get(LAname,lorentzAngleHandle);
  std::string gainLabel = conf_.getParameter<std::string>("Gain");
  iSetup.get<SiStripGainSimRcd>().get(gainLabel,gainHandle);
  iSetup.get<SiStripNoisesRcd>().get(noiseHandle);
  iSetup.get<SiStripThresholdRcd>().get(thresholdHandle);
  iSetup.get<SiStripPedestalsRcd>().get(pedestalHandle);
  iSetup.get<SiStripBadChannelRcd>().get(deadChannelHandle);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  iSetup.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  theDigiAlgo->setParticleDataTable(&*pdt);

  // Step B: LOOP on StripGeomDetUnit
  theDigiVector.reserve(10000);
  theDigiVector.clear();
  theDigiLinkVector.reserve(10000);
  theDigiLinkVector.clear();

  for(TrackingGeometry::DetUnitContainer::const_iterator iu = pDD->detUnits().begin(); iu != pDD->detUnits().end(); iu ++){
    if(useConfFromDB){
      //apply the cable map _before_ digitization: consider only the detis that are connected 
      if(theDetIdList.find((*iu)->geographicalId().rawId())==theDetIdList.end())
        continue;
    }
    GlobalVector bfield=pSetup->inTesla((*iu)->surface().position());
    StripGeomDetUnit* sgd = dynamic_cast<StripGeomDetUnit*>((*iu));
    if (sgd != 0){
      edm::DetSet<SiStripDigi> collectorZS((*iu)->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorRaw((*iu)->geographicalId().rawId());
      edm::DetSet<StripDigiSimLink> linkcollector((*iu)->geographicalId().rawId());
      float langle = (lorentzAngleHandle.isValid()) ? lorentzAngleHandle->getLorentzAngle((*iu)->geographicalId().rawId()) : 0.;
      theDigiAlgo->run(collectorZS,collectorRaw,SimHitMap[(*iu)->geographicalId().rawId()],sgd,bfield,langle,
	 	       gainHandle,thresholdHandle,noiseHandle,pedestalHandle, deadChannelHandle, tTopo);
      if(zeroSuppression){
        if(collectorZS.data.size()>0){
          theDigiVector.push_back(collectorZS);
          if(SimHitMap[(*iu)->geographicalId().rawId()].size()>0){
            linkcollector.data = theDigiAlgo->make_link();
            if(linkcollector.data.size()>0)
              theDigiLinkVector.push_back(linkcollector);
          }
        }
      }else{
        if(collectorRaw.data.size()>0){
          theRawDigiVector.push_back(collectorRaw);
          if(SimHitMap[(*iu)->geographicalId().rawId()].size()>0){
            linkcollector.data = theDigiAlgo->make_link();
            if(linkcollector.data.size()>0)
              theDigiLinkVector.push_back(linkcollector);
          }
        }
      }
    }
  }

  // Step C: create output collection
  std::auto_ptr<edm::DetSetVector<StripDigiSimLink> > outputlink(new edm::DetSetVector<StripDigiSimLink>(theDigiLinkVector));
  // Step D: write output to file
  iEvent.put(outputlink);
}
