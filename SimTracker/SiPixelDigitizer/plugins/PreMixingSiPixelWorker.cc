#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"  // not really needed
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"

#include "CLHEP/Random/RandFlat.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

#include "SiPixelDigitizerAlgorithm.h"

#include <map>
#include <memory>

class PreMixingSiPixelWorker : public PreMixingWorker {
public:
  PreMixingSiPixelWorker(const edm::ParameterSet& ps, edm::ProducesCollector, edm::ConsumesCollector&& iC);
  ~PreMixingSiPixelWorker() override = default;

  void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
  void addSignals(edm::Event const& e, edm::EventSetup const& es) override;
  void addPileups(PileUpEventPrincipal const&, edm::EventSetup const& es) override;
  void put(edm::Event& e, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bs) override;

private:
  edm::InputTag pixeldigi_collectionSig_;   // secondary name given to collection of SiPixel digis
  edm::InputTag pixeldigi_collectionPile_;  // secondary name given to collection of SiPixel digis
  std::string PixelDigiCollectionDM_;       // secondary name to be given to new SiPixel digis

  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> PixelDigiToken_;   // Token to retrieve information
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> PixelDigiPToken_;  // Token to retrieve information

  edm::ESHandle<TrackerGeometry> pDD;

  SiPixelDigitizerAlgorithm digitizer_;

  //
  // Internal typedefs

  typedef int Amplitude;
  typedef std::map<int, Amplitude, std::less<int>> signal_map_type;  // from Digi.Skel.
  typedef std::map<uint32_t, signal_map_type> signalMaps;

  typedef std::multimap<int, PixelDigi>
      OneDetectorMap;  // maps by pixel ID for later combination - can have duplicate pixels
  typedef std::map<uint32_t, OneDetectorMap> SiGlobalIndex;  // map to all data for each detector ID

  SiGlobalIndex SiHitStorage_;

  const std::string geometryType_;

  bool firstInitializeEvent_ = true;
  bool firstFinalizeEvent_ = true;
};

// Constructor
PreMixingSiPixelWorker::PreMixingSiPixelWorker(const edm::ParameterSet& ps,
                                               edm::ProducesCollector producesCollector,
                                               edm::ConsumesCollector&& iC)
    : digitizer_(ps), geometryType_(ps.getParameter<std::string>("PixGeometryType")) {
  // declare the products to produce

  pixeldigi_collectionSig_ = ps.getParameter<edm::InputTag>("pixeldigiCollectionSig");
  pixeldigi_collectionPile_ = ps.getParameter<edm::InputTag>("pixeldigiCollectionPile");
  PixelDigiCollectionDM_ = ps.getParameter<std::string>("PixelDigiCollectionDM");

  PixelDigiToken_ = iC.consumes<edm::DetSetVector<PixelDigi>>(pixeldigi_collectionSig_);
  PixelDigiPToken_ = iC.consumes<edm::DetSetVector<PixelDigi>>(pixeldigi_collectionPile_);

  producesCollector.produces<edm::DetSetVector<PixelDigi>>(PixelDigiCollectionDM_);
  producesCollector.produces<PixelFEDChannelCollection>(PixelDigiCollectionDM_);

  // clear local storage for this event
  SiHitStorage_.clear();
}

// Need an event initialization

void PreMixingSiPixelWorker::initializeEvent(edm::Event const& e, edm::EventSetup const& iSetup) {
  iSetup.get<TrackerDigiGeometryRecord>().get(geometryType_, pDD);
  if (firstInitializeEvent_) {
    digitizer_.init(iSetup);
    firstInitializeEvent_ = false;
  }
  digitizer_.initializeEvent();
}

void PreMixingSiPixelWorker::addSignals(edm::Event const& e, edm::EventSetup const& es) {
  // fill in maps of hits

  LogDebug("PreMixingSiPixelWorker") << "===============> adding MC signals for " << e.id();

  edm::Handle<edm::DetSetVector<PixelDigi>> input;

  if (e.getByToken(PixelDigiToken_, input)) {
    //loop on all detsets (detectorIDs) inside the input collection
    edm::DetSetVector<PixelDigi>::const_iterator DSViter = input->begin();
    for (; DSViter != input->end(); DSViter++) {
#ifdef DEBUG
      LogDebug("PreMixingSiPixelWorker") << "Processing DetID " << DSViter->id;
#endif

      uint32_t detID = DSViter->id;
      edm::DetSet<PixelDigi>::const_iterator begin = (DSViter->data).begin();
      edm::DetSet<PixelDigi>::const_iterator end = (DSViter->data).end();
      edm::DetSet<PixelDigi>::const_iterator icopy;

      OneDetectorMap LocalMap;

      for (icopy = begin; icopy != end; icopy++) {
        LocalMap.insert(OneDetectorMap::value_type((icopy->channel()), *icopy));
      }

      SiHitStorage_.insert(SiGlobalIndex::value_type(detID, LocalMap));
    }
  }
}  // end of addSiPixelSignals

void PreMixingSiPixelWorker::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& es) {
  LogDebug("PreMixingSiPixelWorker") << "\n===============> adding pileups from event  " << pep.principal().id()
                                     << " for bunchcrossing " << pep.bunchCrossing();

  // fill in maps of hits; same code as addSignals, except now applied to the pileup events

  edm::Handle<edm::DetSetVector<PixelDigi>> inputHandle;
  pep.getByLabel(pixeldigi_collectionPile_, inputHandle);

  if (inputHandle.isValid()) {
    const auto& input = *inputHandle;

    //loop on all detsets (detectorIDs) inside the input collection
    edm::DetSetVector<PixelDigi>::const_iterator DSViter = input.begin();
    for (; DSViter != input.end(); DSViter++) {
#ifdef DEBUG
      LogDebug("PreMixingSiPixelWorker") << "Pileups: Processing DetID " << DSViter->id;
#endif

      uint32_t detID = DSViter->id;
      edm::DetSet<PixelDigi>::const_iterator begin = (DSViter->data).begin();
      edm::DetSet<PixelDigi>::const_iterator end = (DSViter->data).end();
      edm::DetSet<PixelDigi>::const_iterator icopy;

      // find correct local map (or new one) for this detector ID

      SiGlobalIndex::const_iterator itest;

      itest = SiHitStorage_.find(detID);

      if (itest != SiHitStorage_.end()) {  // this detID already has hits, add to existing map

        OneDetectorMap LocalMap = itest->second;

        // fill in local map with extra channels
        for (icopy = begin; icopy != end; icopy++) {
          LocalMap.insert(OneDetectorMap::value_type((icopy->channel()), *icopy));
        }

        SiHitStorage_[detID] = LocalMap;

      } else {  // fill local storage with this information, put in global collection

        OneDetectorMap LocalMap;

        for (icopy = begin; icopy != end; icopy++) {
          LocalMap.insert(OneDetectorMap::value_type((icopy->channel()), *icopy));
        }

        SiHitStorage_.insert(SiGlobalIndex::value_type(detID, LocalMap));
      }
    }
  }
}

void PreMixingSiPixelWorker::put(edm::Event& e,
                                 edm::EventSetup const& iSetup,
                                 std::vector<PileupSummaryInfo> const& ps,
                                 int bs) {
  // collection of Digis to put in the event

  std::vector<edm::DetSet<PixelDigi>> vPixelDigi;

  // loop through our collection of detectors, merging hits and putting new ones in the output
  signalMaps signal;

  // big loop over Detector IDs:

  for (SiGlobalIndex::const_iterator IDet = SiHitStorage_.begin(); IDet != SiHitStorage_.end(); IDet++) {
    uint32_t detID = IDet->first;

    OneDetectorMap LocalMap = IDet->second;

    signal_map_type Signals;
    Signals.clear();

    //counter variables
    int formerPixel = -1;
    int currentPixel;
    int ADCSum = 0;

    OneDetectorMap::const_iterator iLocalchk;

    for (OneDetectorMap::const_iterator iLocal = LocalMap.begin(); iLocal != LocalMap.end(); ++iLocal) {
      currentPixel = iLocal->first;

      if (currentPixel == formerPixel) {  // we have to add these digis together
        ADCSum += (iLocal->second).adc();
      } else {
        if (formerPixel != -1) {  // ADC info stolen from SiStrips...
          if (ADCSum > 511)
            ADCSum = 255;
          else if (ADCSum > 253 && ADCSum < 512)
            ADCSum = 254;

          Signals.insert(std::make_pair(formerPixel, ADCSum));
        }
        // save pointers for next iteration
        formerPixel = currentPixel;
        ADCSum = (iLocal->second).adc();
      }

      iLocalchk = iLocal;
      if ((++iLocalchk) == LocalMap.end()) {  //make sure not to lose the last one
        if (ADCSum > 511)
          ADCSum = 255;
        else if (ADCSum > 253 && ADCSum < 512)
          ADCSum = 254;
        Signals.insert(std::make_pair(formerPixel, ADCSum));
      }

    }  // end of loop over one detector

    // stick this into the global vector of detector info
    signal.insert(std::make_pair(detID, Signals));

  }  // end of big loop over all detector IDs

  // put the collection of digis in the event
  edm::LogInfo("PreMixingSiPixelWorker") << "total # Merged Pixels: " << signal.size();

  std::vector<edm::DetSet<PixelDigi>> theDigiVector;

  // Load inefficiency constants (1st pass), set pileup information.
  if (firstFinalizeEvent_) {
    digitizer_.init_DynIneffDB(iSetup, bs);
    firstFinalizeEvent_ = false;
  }

  digitizer_.calculateInstlumiFactor(ps, bs);
  digitizer_.setSimAccumulator(signal);

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  edm::ESHandle<TrackerTopology> tTopoHand;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology* tTopo = tTopoHand.product();

  if (digitizer_.killBadFEDChannels()) {
    std::unique_ptr<PixelFEDChannelCollection> PixelFEDChannelCollection_ = digitizer_.chooseScenario(ps, engine);
    if (PixelFEDChannelCollection_ == nullptr) {
      throw cms::Exception("NullPointerError") << "PixelFEDChannelCollection not set in chooseScenario function.\n";
    }
    e.put(std::move(PixelFEDChannelCollection_), PixelDigiCollectionDM_);
  }

  for (const auto& iu : pDD->detUnits()) {
    if (iu->type().isTrackerPixel()) {
      edm::DetSet<PixelDigi> collector(iu->geographicalId().rawId());
      edm::DetSet<PixelDigiSimLink> linkcollector(
          iu->geographicalId().rawId());  // ignored as DigiSimLinks are combined separately

      digitizer_.digitize(dynamic_cast<const PixelGeomDetUnit*>(iu), collector.data, linkcollector.data, tTopo, engine);
      if (!collector.data.empty()) {
        theDigiVector.push_back(std::move(collector));
      }
    }
  }

  e.put(std::make_unique<edm::DetSetVector<PixelDigi>>(theDigiVector), PixelDigiCollectionDM_);

  // clear local storage for this event
  SiHitStorage_.clear();
}

DEFINE_PREMIXING_WORKER(PreMixingSiPixelWorker);
