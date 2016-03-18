//
//
// Package:    Phase2TrackerDigitizer
// Class:      Phase2TrackerDigitizer
// 
// *\class SiPhase2TrackerDigitizer Phase2TrackerDigitizer.cc SimTracker/SiPhase2Digitizer/src/Phase2TrackerDigitizer.cc
//
// Author: Suchandra Dutta, Suvankar Roy Chowdhury, Subir Sarkar
// Date: January 29, 2016
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//

#include <memory>
#include <set>
#include <iostream>
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizer.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/plugins/SSDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/plugins/PSSDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/plugins/PSPDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/plugins/PixelDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/plugins/DigitizerUtility.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigiCollection.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

// Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"

namespace cms
{

  Phase2TrackerDigitizer::Phase2TrackerDigitizer(const edm::ParameterSet& iConfig, edm::stream::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
    first_(true),
    hitsProducer_(iConfig.getParameter<std::string>("hitsProducer")),
    trackerContainers_(iConfig.getParameter<std::vector<std::string> >("ROUList")),
    geometryType_(iConfig.getParameter<std::string>("GeometryType")),
    iconfig_(iConfig)
  {
    //edm::LogInfo("Phase2TrackerDigitizer") << "Initialize Digitizer Algorithms";
    const std::string alias1("simSiPixelDigis"); 
    mixMod.produces<edm::DetSetVector<PixelDigi> >("Pixel").setBranchAlias(alias1);
    mixMod.produces<edm::DetSetVector<PixelDigiSimLink> >("Pixel").setBranchAlias(alias1);

    const std::string alias2("simSiTrackerDigis"); 
    mixMod.produces<edm::DetSetVector<Phase2TrackerDigi> >("Tracker").setBranchAlias(alias2);
    mixMod.produces<edm::DetSetVector<PixelDigiSimLink> >("Tracker").setBranchAlias(alias2);

  }

  void Phase2TrackerDigitizer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& iSetup) {
    edm::Service<edm::RandomNumberGenerator> rng;
    if (!rng.isAvailable()) {
      throw cms::Exception("Configuration")
        << "Phase2TrackerDigitizer requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
    }
    rndEngine_ = &(rng->getEngine(lumi.index()));

    iSetup.get<IdealMagneticFieldRecord>().get(pSetup_);
    iSetup.get<TrackerTopologyRcd>().get(tTopoHand);

    
    if (theTkDigiGeomWatcher.check(iSetup)) {
      iSetup.get<TrackerDigiGeometryRecord>().get(geometryType_, pDD_);
      detectorUnits_.clear();
      for (auto const & det_u : pDD_->detUnits()) {
	unsigned int detId_raw = det_u->geographicalId().rawId();
	DetId detId = DetId(detId_raw);
	if (DetId(detId).det() == DetId::Detector::Tracker) {
	  const Phase2TrackerGeomDetUnit* pixdet = dynamic_cast<const Phase2TrackerGeomDetUnit*>(det_u);
	  assert(pixdet);
	  detectorUnits_.insert(std::make_pair(detId_raw, pixdet));
	}
      }
    }
  
    // one type of Digi and DigiSimLink suffices 
    // changes in future: InnerPixel -> Tracker
    // creating algorithm objects and pushing them into the map
    algomap_[AlgorithmType::InnerPixel] = std::unique_ptr<Phase2TrackerDigitizerAlgorithm>(new PixelDigitizerAlgorithm(iconfig_, (*rndEngine_)));
    algomap_[AlgorithmType::PixelinPS]  = std::unique_ptr<Phase2TrackerDigitizerAlgorithm>(new PSPDigitizerAlgorithm(iconfig_, (*rndEngine_)));
    algomap_[AlgorithmType::StripinPS]  = std::unique_ptr<Phase2TrackerDigitizerAlgorithm>(new PSSDigitizerAlgorithm(iconfig_, (*rndEngine_)));
    algomap_[AlgorithmType::TwoStrip]   = std::unique_ptr<Phase2TrackerDigitizerAlgorithm>(new SSDigitizerAlgorithm(iconfig_, (*rndEngine_)));
  }

  void Phase2TrackerDigitizer::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& iSetup) {
    algomap_.clear();
  }
  Phase2TrackerDigitizer::~Phase2TrackerDigitizer() {  
    edm::LogInfo("Phase2TrackerDigitizer") << "Destroying the Digitizer";
  }
  void
  Phase2TrackerDigitizer::accumulatePixelHits(edm::Handle<std::vector<PSimHit> > hSimHits,
				       size_t globalSimHitIndex,const unsigned int tofBin) {
    if (hSimHits.isValid()) {
      std::set<unsigned int> detIds;
      std::vector<PSimHit> const& simHits = *hSimHits.product();
      for (auto it = simHits.begin(), itEnd = simHits.end(); it != itEnd; ++it, ++globalSimHitIndex) {
	unsigned int detId_raw = (*it).detUnitId();
        if (detectorUnits_.find(detId_raw) == detectorUnits_.end()) continue;
	if (detIds.insert(detId_raw).second) {
	  // The insert succeeded, so this detector element has not yet been processed.
	  AlgorithmType algotype = getAlgoType(detId_raw);
	  const Phase2TrackerGeomDetUnit* phase2det = detectorUnits_[detId_raw];
	  // access to magnetic field in global coordinates
	  GlobalVector bfield = pSetup_->inTesla(phase2det->surface().position());
	  LogDebug("PixelDigitizer") << "B-field(T) at " << phase2det->surface().position() << "(cm): " 
	         		     << pSetup_->inTesla(phase2det->surface().position());
	  if (algomap_.find(algotype) != algomap_.end()) 
	    algomap_[algotype]->accumulateSimHits(it, itEnd, globalSimHitIndex, tofBin, phase2det, bfield);
	  else
	    edm::LogInfo("Phase2TrackerDigitizer") << "Unsupported algorithm: ";
	}
      }
    }
  }
  
  void
  Phase2TrackerDigitizer::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
    
    // Must initialize all the algorithms
    for (auto const & el : algomap_) {
      if (first_) el.second->init(iSetup); 
      el.second->initializeEvent(); 
    }
    first_ = false;
    // Make sure that the first crossing processed starts indexing the sim hits from zero.
    // This variable is used so that the sim hits from all crossing frames have sequential
    // indices used to create the digi-sim link (if configured to do so) rather than starting
    // from zero for each crossing.
    crossingSimHitIndexOffset_.clear();
  }  
  void 
  Phase2TrackerDigitizer::accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    accumulate_local<edm::Event>(iEvent, iSetup);
  }

  void  
  Phase2TrackerDigitizer::accumulate(PileUpEventPrincipal const& iEvent, edm::EventSetup const& iSetup, edm::StreamID const& ) {
    accumulate_local<PileUpEventPrincipal>(iEvent, iSetup);
  }

  template <class T>
  void Phase2TrackerDigitizer::accumulate_local(T const& iEvent, edm::EventSetup const& iSetup) {
    for (auto const & v : trackerContainers_) {
      edm::Handle<std::vector<PSimHit> > simHits;
      edm::InputTag tag(hitsProducer_, v);
      iEvent.getByLabel(tag, simHits);

      //edm::EDGetTokenT< std::vector<PSimHit> > simHitToken_(consumes< std::vector<PSimHit>(tag));
      //iEvent.getByToken(simHitToken_, simHits);

      unsigned int tofBin = PixelDigiSimLink::LowTof;
      if (v.find(std::string("HighTof")) != std::string::npos) tofBin = PixelDigiSimLink::HighTof;
      accumulatePixelHits(simHits, crossingSimHitIndexOffset_[tag.encode()], tofBin);
      // Now that the hits have been processed, I'll add the amount of hits in this crossing on to
      // the global counter. Next time accumulateStripHits() is called it will count the sim hits
      // as though they were on the end of this collection.
      // Note that this is only used for creating digi-sim links (if configured to do so).
      if (simHits.isValid()) crossingSimHitIndexOffset_[tag.encode()] += simHits->size();
     }
  }
  void Phase2TrackerDigitizer::finalizeEvent(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const bool isOuterTrackerReadoutAnalog = iconfig_.getParameter<bool>("isOTreadoutAnalog");
    //Decide if we want analog readout for Outer Tracker.
    addPixelCollection(iEvent, iSetup, isOuterTrackerReadoutAnalog);
    if(!isOuterTrackerReadoutAnalog)
      addOuterTrackerCollection(iEvent, iSetup);
  }
  void Phase2TrackerDigitizer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  }
  Phase2TrackerDigitizer::AlgorithmType Phase2TrackerDigitizer::getAlgoType(unsigned int detId_raw) {
    DetId detId(detId_raw); 

    AlgorithmType algotype = AlgorithmType::Unknown;
    TrackerGeometry::ModuleType mType = pDD_->getDetectorType(detId);    
    switch(mType){

    case TrackerGeometry::ModuleType::Ph2PXB:
      algotype = AlgorithmType::InnerPixel;
      break;
    case TrackerGeometry::ModuleType::Ph2PXF:
      algotype = AlgorithmType::InnerPixel;
      break;
    case TrackerGeometry::ModuleType::Ph2PSP:
      algotype = AlgorithmType::PixelinPS;
      break;
    case TrackerGeometry::ModuleType::Ph2PSS:
      algotype = AlgorithmType::StripinPS;
      break;
    case TrackerGeometry::ModuleType::Ph2SS:
      algotype = AlgorithmType::TwoStrip;
      break;
    default:
      edm::LogError("Phase2TrackerDigitizer")<<"ERROR - Wrong Detector Type, No Algorithm available ";
    }
 
    return algotype;
  }
  void Phase2TrackerDigitizer::addPixelCollection(edm::Event& iEvent, const edm::EventSetup& iSetup, const bool ot_analog) {
    const TrackerTopology* tTopo = tTopoHand.product();
    std::vector<edm::DetSet<PixelDigi> > digiVector;
    std::vector<edm::DetSet<PixelDigiSimLink> > digiLinkVector;
    for (auto const & det_u : pDD_->detUnits()) {
      DetId detId_raw = DetId(det_u->geographicalId().rawId());
      AlgorithmType algotype = getAlgoType(detId_raw);
      if (algomap_.find(algotype) == algomap_.end()) continue;

      //Decide if we want analog readout for Outer Tracker.
      if( !ot_analog && algotype != AlgorithmType::InnerPixel) continue;
      std::map<int, DigitizerUtility::DigiSimInfo> digi_map;
      algomap_[algotype]->digitize(dynamic_cast<const Phase2TrackerGeomDetUnit*>(det_u),
                                   digi_map,tTopo);
      edm::DetSet<PixelDigi> collector(det_u->geographicalId().rawId());
      edm::DetSet<PixelDigiSimLink> linkcollector(det_u->geographicalId().rawId());
      for (auto const & digi_p : digi_map) {
	DigitizerUtility::DigiSimInfo info = digi_p.second;  
	std::pair<int,int> ip = PixelDigi::channelToPixel(digi_p.first);
	collector.data.emplace_back(ip.first, ip.second, info.sig_tot);
        for (auto const & tk_p : info.track_map) {
	  linkcollector.data.emplace_back(digi_p.first, tk_p.first, info.hit_counter, info.tof_bin, info.event_id, tk_p.second);
	}
      }  	
      if (collector.data.size() > 0) digiVector.push_back(std::move(collector));	  
      if (linkcollector.data.size() > 0) digiLinkVector.push_back(std::move(linkcollector));
    } 
    
    // Step C: create collection with the cache vector of DetSet 
    std::auto_ptr<edm::DetSetVector<PixelDigi> > 
      output(new edm::DetSetVector<PixelDigi>(digiVector));
    std::auto_ptr<edm::DetSetVector<PixelDigiSimLink> > 
      outputlink(new edm::DetSetVector<PixelDigiSimLink>(digiLinkVector));
    
    // Step D: write output to file 
    iEvent.put(output, "Pixel");
    iEvent.put(outputlink, "Pixel");
  }
  void Phase2TrackerDigitizer::addOuterTrackerCollection(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const TrackerTopology* tTopo = tTopoHand.product();
    std::vector<edm::DetSet<Phase2TrackerDigi> > digiVector;
    std::vector<edm::DetSet<PixelDigiSimLink> > digiLinkVector;
    for (auto const & det_u : pDD_->detUnits()) {
      DetId detId_raw = DetId(det_u->geographicalId().rawId());
      AlgorithmType algotype = getAlgoType(detId_raw);

      if (algomap_.find(algotype) == algomap_.end() || algotype == AlgorithmType::InnerPixel) continue;

      std::map<int, DigitizerUtility::DigiSimInfo> digi_map;
      algomap_[algotype]->digitize(dynamic_cast<const Phase2TrackerGeomDetUnit*>(det_u),
				   digi_map, tTopo);
      edm::DetSet<Phase2TrackerDigi> collector(det_u->geographicalId().rawId());
      edm::DetSet<PixelDigiSimLink> linkcollector(det_u->geographicalId().rawId());

      for (auto const & digi_p : digi_map) {
	DigitizerUtility::DigiSimInfo info = digi_p.second;  
	std::pair<int,int> ip = Phase2TrackerDigi::channelToPixel(digi_p.first);
	collector.data.emplace_back(ip.first, ip.second);
        for (auto const & track_p : info.track_map) {
	  linkcollector.data.emplace_back(digi_p.first, track_p.first, info.hit_counter, info.tof_bin, info.event_id, track_p.second);
	}
      }  	
	
      if (collector.data.size() > 0) digiVector.push_back(std::move(collector));	  
      if (linkcollector.data.size() > 0) digiLinkVector.push_back(std::move(linkcollector));
    } 
    
    // Step C: create collection with the cache vector of DetSet 
    std::auto_ptr<edm::DetSetVector<Phase2TrackerDigi> > 
      output(new edm::DetSetVector<Phase2TrackerDigi>(digiVector));
    std::auto_ptr<edm::DetSetVector<PixelDigiSimLink> > 
      outputlink(new edm::DetSetVector<PixelDigiSimLink>(digiLinkVector));
    
    // Step D: write output to file 
    iEvent.put(output, "Tracker");
    iEvent.put(outputlink, "Tracker");
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"

using cms::Phase2TrackerDigitizer;
DEFINE_DIGI_ACCUMULATOR(Phase2TrackerDigitizer);
