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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
//needed for the geometry:
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
//needed for the magnetic field:
#include "MagneticField/Engine/interface/MagneticField.h"

//Data Base infromations
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandFlat.h"

SiStripDigitizer::SiStripDigitizer(const edm::ParameterSet& conf,
                                   edm::ProducesCollector producesCollector,
                                   edm::ConsumesCollector& iC)
    : hitsProducer(conf.getParameter<std::string>("hitsProducer")),
      trackerContainers(conf.getParameter<std::vector<std::string>>("ROUList")),
      ZSDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("ZSDigi")),
      SCDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("SCDigi")),
      VRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("VRDigi")),
      PRDigi(conf.getParameter<edm::ParameterSet>("DigiModeList").getParameter<std::string>("PRDigi")),
      useConfFromDB(conf.getParameter<bool>("TrackerConfigurationFromDB")),
      zeroSuppression(conf.getParameter<bool>("ZeroSuppression")),
      makeDigiSimLinks_(conf.getUntrackedParameter<bool>("makeDigiSimLinks", false)),
      includeAPVSimulation_(conf.getParameter<bool>("includeAPVSimulation")),
      fracOfEventsToSimAPV_(conf.getParameter<double>("fracOfEventsToSimAPV")),
      tTopoToken_(iC.esConsumes()),
      pDDToken_(iC.esConsumes(edm::ESInputTag("", conf.getParameter<std::string>("GeometryType")))),
      pSetupToken_(iC.esConsumes()),
      gainToken_(iC.esConsumes(edm::ESInputTag("", conf.getParameter<std::string>("Gain")))),
      noiseToken_(iC.esConsumes()),
      thresholdToken_(iC.esConsumes()),
      pedestalToken_(iC.esConsumes()),
      deadChannelToken_(iC.esConsumes()) {
  if (useConfFromDB) {
    detCablingToken_ = iC.esConsumes();
  }
  if (includeAPVSimulation_) {
    apvSimulationParametersToken_ = iC.esConsumes();
  }

  const std::string alias("simSiStripDigis");

  producesCollector.produces<edm::DetSetVector<SiStripDigi>>(ZSDigi).setBranchAlias(ZSDigi);
  producesCollector.produces<edm::DetSetVector<SiStripRawDigi>>(SCDigi).setBranchAlias(alias + SCDigi);
  producesCollector.produces<edm::DetSetVector<SiStripRawDigi>>(VRDigi).setBranchAlias(alias + VRDigi);
  producesCollector.produces<edm::DetSetVector<SiStripRawDigi>>("StripAmplitudes")
      .setBranchAlias(alias + "StripAmplitudes");
  producesCollector.produces<edm::DetSetVector<SiStripRawDigi>>("StripAmplitudesPostAPV")
      .setBranchAlias(alias + "StripAmplitudesPostAPV");
  producesCollector.produces<edm::DetSetVector<SiStripRawDigi>>("StripAPVBaselines")
      .setBranchAlias(alias + "StripAPVBaselines");
  producesCollector.produces<edm::DetSetVector<SiStripRawDigi>>(PRDigi).setBranchAlias(alias + PRDigi);
  producesCollector.produces<edm::DetSetVector<StripDigiSimLink>>().setBranchAlias(alias + "siStripDigiSimLink");
  producesCollector.produces<bool>("SimulatedAPVDynamicGain").setBranchAlias(alias + "SimulatedAPVDynamicGain");
  producesCollector.produces<std::vector<std::pair<int, std::bitset<6>>>>("AffectedAPVList")
      .setBranchAlias(alias + "AffectedAPV");
  for (auto const& trackerContainer : trackerContainers) {
    edm::InputTag tag(hitsProducer, trackerContainer);
    iC.consumes<std::vector<PSimHit>>(edm::InputTag(hitsProducer, trackerContainer));
  }
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "SiStripDigitizer requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  }
  theDigiAlgo = std::make_unique<SiStripDigitizerAlgorithm>(conf, iC);
}

// Virtual destructor needed.
SiStripDigitizer::~SiStripDigitizer() {}

void SiStripDigitizer::accumulateStripHits(edm::Handle<std::vector<PSimHit>> hSimHits,
                                           const TrackerTopology* tTopo,
                                           size_t globalSimHitIndex,
                                           const unsigned int tofBin) {
  // globalSimHitIndex is the index the sim hit will have when it is put in a collection
  // of sim hits for all crossings. This is only used when creating digi-sim links if
  // configured to do so.

  if (hSimHits.isValid()) {
    std::set<unsigned int> detIds;
    std::vector<PSimHit> const& simHits = *hSimHits.product();
    for (std::vector<PSimHit>::const_iterator it = simHits.begin(), itEnd = simHits.end(); it != itEnd;
         ++it, ++globalSimHitIndex) {
      unsigned int detId = (*it).detUnitId();
      if (detIds.insert(detId).second) {
        // The insert succeeded, so this detector element has not yet been processed.
        assert(detectorUnits[detId]);
        if (detectorUnits[detId]->type().isTrackerStrip()) {  // this test can be removed and replaced by stripdet!=0
          auto stripdet = detectorUnits[detId];
          //access to magnetic field in global coordinates
          GlobalVector bfield = pSetup->inTesla(stripdet->surface().position());
          LogDebug("Digitizer ") << "B-field(T) at " << stripdet->surface().position()
                                 << "(cm): " << pSetup->inTesla(stripdet->surface().position());
          theDigiAlgo->accumulateSimHits(it, itEnd, globalSimHitIndex, tofBin, stripdet, bfield, tTopo, randomEngine_);
        }
      }
    }  // end of loop over sim hits
  }
}

// Functions that gets called by framework every event
void SiStripDigitizer::accumulate(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  //Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &iSetup.getData(tTopoToken_);

  // Step A: Get Inputs
  for (auto const& trackerContainer : trackerContainers) {
    edm::Handle<std::vector<PSimHit>> simHits;
    edm::InputTag tag(hitsProducer, trackerContainer);
    unsigned int tofBin = StripDigiSimLink::LowTof;
    if (trackerContainer.find(std::string("HighTof")) != std::string::npos)
      tofBin = StripDigiSimLink::HighTof;

    iEvent.getByLabel(tag, simHits);
    accumulateStripHits(simHits, tTopo, crossingSimHitIndexOffset_[tag.encode()], tofBin);
    // Now that the hits have been processed, I'll add the amount of hits in this crossing on to
    // the global counter. Next time accumulateStripHits() is called it will count the sim hits
    // as though they were on the end of this collection.
    // Note that this is only used for creating digi-sim links (if configured to do so).
    if (simHits.isValid())
      crossingSimHitIndexOffset_[tag.encode()] += simHits->size();
  }
}

void SiStripDigitizer::accumulate(PileUpEventPrincipal const& iEvent,
                                  edm::EventSetup const& iSetup,
                                  edm::StreamID const& streamID) {
  const TrackerTopology* tTopo = &iSetup.getData(tTopoToken_);

  //Re-compute luminosity for accumulation for HIP effects
  theDigiAlgo->calculateInstlumiScale(PileupInfo_.get());

  // Step A: Get Inputs
  for (auto const& trackerContainer : trackerContainers) {
    edm::Handle<std::vector<PSimHit>> simHits;
    edm::InputTag tag(hitsProducer, trackerContainer);
    unsigned int tofBin = StripDigiSimLink::LowTof;
    if (trackerContainer.find(std::string("HighTof")) != std::string::npos)
      tofBin = StripDigiSimLink::HighTof;

    iEvent.getByLabel(tag, simHits);
    accumulateStripHits(simHits, tTopo, crossingSimHitIndexOffset_[tag.encode()], tofBin);
    // Now that the hits have been processed, I'll add the amount of hits in this crossing on to
    // the global counter. Next time accumulateStripHits() is called it will count the sim hits
    // as though they were on the end of this collection.
    // Note that this is only used for creating digi-sim links (if configured to do so).
    if (simHits.isValid())
      crossingSimHitIndexOffset_[tag.encode()] += simHits->size();
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

  if (useConfFromDB) {
    iSetup.getData(detCablingToken_).addConnected(theDetIdList);
  }

  // Cache random number engine
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(iEvent.streamID());

  theDigiAlgo->initializeEvent(iSetup);

  pDD = &iSetup.getData(pDDToken_);
  pSetup = &iSetup.getData(pSetupToken_);

  // We only need to clear and (re)fill detectorUnits when the geometry type IOV changes.
  auto ddCache = iSetup.get<TrackerDigiGeometryRecord>().cacheIdentifier();
  auto deadChannelCache = iSetup.get<SiStripBadChannelRcd>().cacheIdentifier();
  if (ddCache != ddCacheID_ or deadChannelCache != deadChannelCacheID_) {
    ddCacheID_ = ddCache;
    deadChannelCacheID_ = deadChannelCache;
    detectorUnits.clear();

    auto const& deadChannel = iSetup.getData(deadChannelToken_);
    for (const auto& iu : pDD->detUnits()) {
      unsigned int detId = iu->geographicalId().rawId();
      if (iu->type().isTrackerStrip()) {
        auto stripdet = dynamic_cast<StripGeomDetUnit const*>(iu);
        assert(stripdet != nullptr);
        detectorUnits.insert(std::make_pair(detId, stripdet));
        theDigiAlgo->initializeDetUnit(stripdet, deadChannel);
      }
    }
  }
}

void SiStripDigitizer::finalizeEvent(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  auto const& gain = iSetup.getData(gainToken_);
  auto const& noise = iSetup.getData(noiseToken_);
  auto const& threshold = iSetup.getData(thresholdToken_);
  auto const& pedestal = iSetup.getData(pedestalToken_);
  SiStripApvSimulationParameters const* apvSimulationParameters = nullptr;

  std::unique_ptr<bool> simulateAPVInThisEvent = std::make_unique<bool>(false);
  if (includeAPVSimulation_) {
    if (CLHEP::RandFlat::shoot(randomEngine_) < fracOfEventsToSimAPV_) {
      *simulateAPVInThisEvent = true;
      apvSimulationParameters = &iSetup.getData(apvSimulationParametersToken_);
    }
  }
  std::vector<edm::DetSet<SiStripDigi>> theDigiVector;
  std::vector<edm::DetSet<SiStripRawDigi>> theRawDigiVector;
  std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> theStripAmplitudeVector(new edm::DetSetVector<SiStripRawDigi>());
  std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> theStripAmplitudeVectorPostAPV(
      new edm::DetSetVector<SiStripRawDigi>());
  std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> theStripAPVBaselines(new edm::DetSetVector<SiStripRawDigi>());
  std::unique_ptr<edm::DetSetVector<StripDigiSimLink>> pOutputDigiSimLink(new edm::DetSetVector<StripDigiSimLink>);

  const TrackerTopology* tTopo = &iSetup.getData(tTopoToken_);

  // Step B: LOOP on StripGeomDetUnit
  theDigiVector.reserve(10000);
  theDigiVector.clear();

  for (const auto& iu : pDD->detUnits()) {
    if (useConfFromDB) {
      //apply the cable map _before_ digitization: consider only the detis that are connected
      if (theDetIdList.find(iu->geographicalId().rawId()) == theDetIdList.end())
        continue;
    }
    auto sgd = dynamic_cast<StripGeomDetUnit const*>(iu);
    if (sgd != nullptr) {
      edm::DetSet<SiStripDigi> collectorZS(iu->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorRaw(iu->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorStripAmplitudes(iu->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorStripAmplitudesPostAPV(iu->geographicalId().rawId());
      edm::DetSet<SiStripRawDigi> collectorStripAPVBaselines(iu->geographicalId().rawId());
      edm::DetSet<StripDigiSimLink> collectorLink(iu->geographicalId().rawId());

      unsigned int detID = sgd->geographicalId().rawId();
      DetId detId(detID);

      theDigiAlgo->digitize(collectorZS,
                            collectorRaw,
                            collectorStripAmplitudes,
                            collectorStripAmplitudesPostAPV,
                            collectorStripAPVBaselines,
                            collectorLink,
                            sgd,
                            gain,
                            threshold,
                            noise,
                            pedestal,
                            *simulateAPVInThisEvent,
                            apvSimulationParameters,
                            theAffectedAPVvector,
                            randomEngine_,
                            tTopo);

      if (!collectorStripAmplitudes.data.empty())
        theStripAmplitudeVector->insert(collectorStripAmplitudes);
      if (!collectorStripAmplitudesPostAPV.data.empty())
        theStripAmplitudeVectorPostAPV->insert(collectorStripAmplitudesPostAPV);
      if (!collectorStripAPVBaselines.data.empty())
        theStripAPVBaselines->insert(collectorStripAPVBaselines);

      if (zeroSuppression) {
        if (!collectorZS.data.empty()) {
          theDigiVector.push_back(collectorZS);
          if (!collectorLink.data.empty())
            pOutputDigiSimLink->insert(collectorLink);
        }
      } else {
        if (!collectorRaw.data.empty()) {
          theRawDigiVector.push_back(collectorRaw);
          if (!collectorLink.data.empty())
            pOutputDigiSimLink->insert(collectorLink);
        }
      }
    }
  }
  if (zeroSuppression) {
    // Step C: create output collection
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> output_virginraw(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> output_scopemode(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> output_processedraw(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripDigi>> output(new edm::DetSetVector<SiStripDigi>(theDigiVector));
    std::unique_ptr<std::vector<std::pair<int, std::bitset<6>>>> AffectedAPVList(
        new std::vector<std::pair<int, std::bitset<6>>>(theAffectedAPVvector));

    // Step D: write output to file
    iEvent.put(std::move(output), ZSDigi);
    iEvent.put(std::move(output_scopemode), SCDigi);
    iEvent.put(std::move(output_virginraw), VRDigi);
    iEvent.put(std::move(theStripAmplitudeVector), "StripAmplitudes");
    iEvent.put(std::move(theStripAmplitudeVectorPostAPV), "StripAmplitudesPostAPV");
    iEvent.put(std::move(theStripAPVBaselines), "StripAPVBaselines");
    iEvent.put(std::move(output_processedraw), PRDigi);
    iEvent.put(std::move(AffectedAPVList), "AffectedAPVList");
    iEvent.put(std::move(simulateAPVInThisEvent), "SimulatedAPVDynamicGain");
    if (makeDigiSimLinks_)
      iEvent.put(
          std::move(pOutputDigiSimLink));  // The previous EDProducer didn't name this collection so I won't either
  } else {
    // Step C: create output collection
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> output_virginraw(
        new edm::DetSetVector<SiStripRawDigi>(theRawDigiVector));
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> output_scopemode(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripRawDigi>> output_processedraw(new edm::DetSetVector<SiStripRawDigi>());
    std::unique_ptr<edm::DetSetVector<SiStripDigi>> output(new edm::DetSetVector<SiStripDigi>());

    // Step D: write output to file
    iEvent.put(std::move(output), ZSDigi);
    iEvent.put(std::move(output_scopemode), SCDigi);
    iEvent.put(std::move(output_virginraw), VRDigi);
    iEvent.put(std::move(theStripAmplitudeVector), "StripAmplitudes");
    iEvent.put(std::move(theStripAmplitudeVectorPostAPV), "StripAmplitudesPostAPV");
    iEvent.put(std::move(theStripAPVBaselines), "StripAPVBaselines");
    iEvent.put(std::move(output_processedraw), PRDigi);
    iEvent.put(std::move(simulateAPVInThisEvent), "SimulatedAPVDynamicGain");
    if (makeDigiSimLinks_)
      iEvent.put(
          std::move(pOutputDigiSimLink));  // The previous EDProducer didn't name this collection so I won't either
  }
  randomEngine_ = nullptr;  // to prevent access outside event
}
