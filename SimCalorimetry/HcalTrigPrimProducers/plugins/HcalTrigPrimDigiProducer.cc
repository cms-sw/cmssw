#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"
#include "CondFormats/HcalObjects/interface/HcalTPChannelParameters.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"

#include <algorithm>
#include <vector>

class HcalTrigPrimDigiProducer : public edm::stream::EDProducer<> {
public:
  explicit HcalTrigPrimDigiProducer(const edm::ParameterSet& ps);
  ~HcalTrigPrimDigiProducer() override {}

  /**Produces the EDM products,*/
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  HcalTriggerPrimitiveAlgo theAlgo_;

  /// input tags for HCAL digis
  std::vector<edm::InputTag> inputLabel_;
  std::vector<edm::InputTag> inputUpgradeLabel_;
  // this seems a strange way of doing things
  edm::EDGetTokenT<QIE11DigiCollection> tok_hbhe_up_;
  edm::EDGetTokenT<QIE10DigiCollection> tok_hf_up_;

  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;

  bool overrideDBweightsAndFilterHE_;
  bool overrideDBweightsAndFilterHB_;

  /// input tag for FEDRawDataCollection
  edm::InputTag inputTagFEDRaw_;
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  double MinLongEnergy_, MinShortEnergy_, LongShortSlope_, LongShortOffset_;

  bool runZS_;

  bool runFrontEndFormatError_;

  bool upgrade_;
  bool legacy_;

  bool HFEMB_;
  edm::ParameterSet LongShortCut_;
  edm::ESGetToken<HcalTPGCoder, HcalTPGRecord> tok_tpgCoder_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> tok_tpgTranscoder_;
  edm::ESGetToken<HcalLutMetadata, HcalLutMetadataRcd> tok_lutMetadata_;
  edm::ESGetToken<HcalTrigTowerGeometry, CaloGeometryRecord> tok_trigTowerGeom_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_hcalTopo_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbService_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_dbService_beginRun_;
};

HcalTrigPrimDigiProducer::HcalTrigPrimDigiProducer(const edm::ParameterSet& ps)
    : theAlgo_(ps.getParameter<bool>("peakFilter"),
               ps.getParameter<std::vector<double> >("weights"),
               ps.getParameter<int>("latency"),
               ps.getParameter<uint32_t>("FG_threshold"),
               ps.getParameter<std::vector<uint32_t> >("FG_HF_thresholds"),
               ps.getParameter<uint32_t>("ZS_threshold"),
               ps.getParameter<int>("numberOfSamples"),
               ps.getParameter<int>("numberOfPresamples"),
               ps.getParameter<int>("numberOfFilterPresamplesHBQIE11"),
               ps.getParameter<int>("numberOfFilterPresamplesHEQIE11"),
               ps.getParameter<int>("numberOfSamplesHF"),
               ps.getParameter<int>("numberOfPresamplesHF"),
               ps.getParameter<bool>("useTDCInMinBiasBits"),
               ps.getParameter<uint32_t>("MinSignalThreshold"),
               ps.getParameter<uint32_t>("PMTNoiseThreshold")),
      inputLabel_(ps.getParameter<std::vector<edm::InputTag> >("inputLabel")),
      inputUpgradeLabel_(ps.getParameter<std::vector<edm::InputTag> >("inputUpgradeLabel")),
      inputTagFEDRaw_(ps.getParameter<edm::InputTag>("InputTagFEDRaw")),
      runZS_(ps.getParameter<bool>("RunZS")),
      runFrontEndFormatError_(ps.getParameter<bool>("FrontEndFormatError")) {
  std::vector<bool> upgrades = {
      ps.getParameter<bool>("upgradeHB"), ps.getParameter<bool>("upgradeHE"), ps.getParameter<bool>("upgradeHF")};
  upgrade_ = std::any_of(std::begin(upgrades), std::end(upgrades), [](bool a) { return a; });
  legacy_ = std::any_of(std::begin(upgrades), std::end(upgrades), [](bool a) { return !a; });

  overrideDBweightsAndFilterHE_ = ps.getParameter<bool>("overrideDBweightsAndFilterHE");
  overrideDBweightsAndFilterHB_ = ps.getParameter<bool>("overrideDBweightsAndFilterHB");

  theAlgo_.setWeightsQIE11(ps.getParameter<edm::ParameterSet>("weightsQIE11"));

  if (ps.exists("parameters")) {
    auto pset = ps.getUntrackedParameter<edm::ParameterSet>("parameters");
    theAlgo_.overrideParameters(pset);
  }
  theAlgo_.setUpgradeFlags(upgrades[0], upgrades[1], upgrades[2]);
  theAlgo_.setFixSaturationFlag(ps.getParameter<bool>("applySaturationFix"));

  HFEMB_ = false;
  if (ps.exists("LSConfig")) {
    LongShortCut_ = ps.getUntrackedParameter<edm::ParameterSet>("LSConfig");
    HFEMB_ = LongShortCut_.getParameter<bool>("HcalFeatureHFEMBit");
    MinLongEnergy_ = LongShortCut_.getParameter<double>("Min_Long_Energy");    //minimum long energy
    MinShortEnergy_ = LongShortCut_.getParameter<double>("Min_Short_Energy");  //minimum short energy
    LongShortSlope_ =
        LongShortCut_.getParameter<double>("Long_vrs_Short_Slope");  //slope of the line that cuts are based on
    LongShortOffset_ = LongShortCut_.getParameter<double>("Long_Short_Offset");  //offset of line
  }
  tok_tpgCoder_ = esConsumes<HcalTPGCoder, HcalTPGRecord>();
  tok_tpgTranscoder_ = esConsumes<CaloTPGTranscoder, CaloTPGRecord>();
  tok_lutMetadata_ = esConsumes<HcalLutMetadata, HcalLutMetadataRcd>();
  tok_trigTowerGeom_ = esConsumes<HcalTrigTowerGeometry, CaloGeometryRecord>();
  tok_hcalTopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();

  // register for data access
  if (runFrontEndFormatError_) {
    tok_raw_ = consumes<FEDRawDataCollection>(inputTagFEDRaw_);
  }

  if (legacy_) {
    tok_hbhe_ = consumes<HBHEDigiCollection>(inputLabel_[0]);
    tok_hf_ = consumes<HFDigiCollection>(inputLabel_[1]);
  }

  if (upgrade_) {
    tok_hbhe_up_ = consumes<QIE11DigiCollection>(inputUpgradeLabel_[0]);
    tok_hf_up_ = consumes<QIE10DigiCollection>(inputUpgradeLabel_[1]);
  }
  tok_dbService_ = esConsumes<HcalDbService, HcalDbRecord>();
  tok_dbService_beginRun_ = esConsumes<HcalDbService, HcalDbRecord, edm::Transition::BeginRun>();
  produces<HcalTrigPrimDigiCollection>();
  theAlgo_.setPeakFinderAlgorithm(ps.getParameter<int>("PeakFinderAlgorithm"));

  edm::ParameterSet hfSS = ps.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HF");

  theAlgo_.setNCTScaleShift(hfSS.getParameter<int>("NCTShift"));
  theAlgo_.setRCTScaleShift(hfSS.getParameter<int>("RCTShift"));
}

void HcalTrigPrimDigiProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  edm::ESHandle<HcalDbService> db = eventSetup.getHandle(tok_dbService_beginRun_);
  const HcalTopology* topo = &eventSetup.getData(tok_hcalTopo_);

  const HcalElectronicsMap* emap = db->getHcalMapping();

  int lastHERing = topo->lastHERing();
  int lastHBRing = topo->lastHBRing();

  // First, determine if we should configure for the filter scheme
  // Check the tp version to make this determination
  bool foundHB = false;
  bool foundHE = false;
  bool newHBtp = false;
  bool newHEtp = false;
  std::vector<HcalElectronicsId> vIds = emap->allElectronicsIdTrigger();
  for (std::vector<HcalElectronicsId>::const_iterator eId = vIds.begin(); eId != vIds.end(); eId++) {
    // The first HB or HE id is enough to tell whether to use new scheme in HB or HE
    if (foundHB and foundHE)
      break;

    HcalTrigTowerDetId hcalTTDetId(emap->lookupTrigger(*eId));
    if (hcalTTDetId.null())
      continue;

    int aieta = abs(hcalTTDetId.ieta());
    int tp_version = hcalTTDetId.version();

    if (aieta <= lastHBRing) {
      foundHB = true;
      if (tp_version > 1)
        newHBtp = true;
    } else if (aieta > lastHBRing and aieta < lastHERing) {
      foundHE = true;
      if (tp_version > 1)
        newHEtp = true;
    }
  }

  std::vector<HcalElectronicsId> eIds = emap->allElectronicsIdPrecision();
  for (std::vector<HcalElectronicsId>::const_iterator eId = eIds.begin(); eId != eIds.end(); eId++) {
    HcalGenericDetId gid(emap->lookup(*eId));
    if (gid.null() or (gid.genericSubdet() != HcalGenericDetId::HcalGenBarrel and
                       gid.genericSubdet() != HcalGenericDetId::HcalGenEndcap))
      continue;

    HcalDetId hcalDetId(emap->lookup(*eId));
    if (hcalDetId.null())
      continue;

    int aieta = abs(hcalDetId.ieta());

    // Do not let ieta 29 in the map
    // If the aieta already has a weight in the map, then move on
    if (aieta < lastHERing) {
      // Filter weight represented in fixed point 8 bit
      int fixedPointWeight = db->getHcalTPChannelParameter(hcalDetId)->getauxi1();

      if (aieta <= lastHBRing) {
        // Fix number of filter presamples to one if we are using DB weights
        // Size of filter is already known when using DB weights
        // Weight from DB represented as 8-bit integer
        if (!overrideDBweightsAndFilterHB_) {
          if (newHBtp) {
            theAlgo_.setNumFilterPresamplesHBQIE11(1);
            theAlgo_.setWeightQIE11(aieta, -static_cast<double>(fixedPointWeight) / 256.0);
          } else {
            theAlgo_.setNumFilterPresamplesHBQIE11(0);
            theAlgo_.setWeightQIE11(aieta, 1.0);
          }
        }
      } else if (aieta > lastHBRing) {
        if (!overrideDBweightsAndFilterHE_) {
          if (newHEtp) {
            theAlgo_.setNumFilterPresamplesHEQIE11(1);
            theAlgo_.setWeightQIE11(aieta, -static_cast<double>(fixedPointWeight) / 256.0);
          } else {
            theAlgo_.setNumFilterPresamplesHEQIE11(0);
            theAlgo_.setWeightQIE11(aieta, 1.0);
          }
        }
      }
    }
  }
}

void HcalTrigPrimDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& eventSetup) {
  // Step A: get the conditions, for the decoding
  edm::ESHandle<HcalTPGCoder> inputCoder = eventSetup.getHandle(tok_tpgCoder_);

  edm::ESHandle<CaloTPGTranscoder> outTranscoder = eventSetup.getHandle(tok_tpgTranscoder_);

  edm::ESHandle<HcalLutMetadata> lutMetadata = eventSetup.getHandle(tok_lutMetadata_);
  float rctlsb = lutMetadata->getRctLsb();

  edm::ESHandle<HcalTrigTowerGeometry> pG = eventSetup.getHandle(tok_trigTowerGeom_);

  // Step B: Create empty output
  std::unique_ptr<HcalTrigPrimDigiCollection> result(new HcalTrigPrimDigiCollection());

  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<HFDigiCollection> hfDigis;

  edm::Handle<QIE11DigiCollection> hbheUpDigis;
  edm::Handle<QIE10DigiCollection> hfUpDigis;

  if (legacy_) {
    iEvent.getByToken(tok_hbhe_, hbheDigis);
    iEvent.getByToken(tok_hf_, hfDigis);

    // protect here against missing input collections
    // there is no protection in HcalTriggerPrimitiveAlgo

    if (!hbheDigis.isValid() and legacy_) {
      edm::LogInfo("HcalTrigPrimDigiProducer") << "\nWarning: HBHEDigiCollection with input tag " << inputLabel_[0]
                                               << "\nrequested in configuration, but not found in the event."
                                               << "\nQuit returning empty product." << std::endl;

      // put empty HcalTrigPrimDigiCollection in the event
      iEvent.put(std::move(result));

      return;
    }

    if (!hfDigis.isValid() and legacy_) {
      edm::LogInfo("HcalTrigPrimDigiProducer") << "\nWarning: HFDigiCollection with input tag " << inputLabel_[1]
                                               << "\nrequested in configuration, but not found in the event."
                                               << "\nQuit returning empty product." << std::endl;

      // put empty HcalTrigPrimDigiCollection in the event
      iEvent.put(std::move(result));

      return;
    }
  }

  if (upgrade_) {
    iEvent.getByToken(tok_hbhe_up_, hbheUpDigis);
    iEvent.getByToken(tok_hf_up_, hfUpDigis);

    if (!hbheUpDigis.isValid() and upgrade_) {
      edm::LogInfo("HcalTrigPrimDigiProducer")
          << "\nWarning: Upgrade HBHEDigiCollection with input tag " << inputUpgradeLabel_[0]
          << "\nrequested in configuration, but not found in the event."
          << "\nQuit returning empty product." << std::endl;

      // put empty HcalTrigPrimDigiCollection in the event
      iEvent.put(std::move(result));

      return;
    }

    if (!hfUpDigis.isValid() and upgrade_) {
      edm::LogInfo("HcalTrigPrimDigiProducer") << "\nWarning: HFDigiCollection with input tag " << inputUpgradeLabel_[1]
                                               << "\nrequested in configuration, but not found in the event."
                                               << "\nQuit returning empty product." << std::endl;

      // put empty HcalTrigPrimDigiCollection in the event
      iEvent.put(std::move(result));

      return;
    }
  }

  edm::ESHandle<HcalDbService> pSetup = eventSetup.getHandle(tok_dbService_);

  HcalFeatureBit* hfembit = nullptr;

  if (HFEMB_) {
    hfembit = new HcalFeatureHFEMBit(MinShortEnergy_,
                                     MinLongEnergy_,
                                     LongShortSlope_,
                                     LongShortOffset_,
                                     *pSetup);  //inputs values that cut will be based on
  }

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  if (legacy_ and not upgrade_) {
    theAlgo_.run(inputCoder.product(),
                 outTranscoder->getHcalCompressor().get(),
                 pSetup.product(),
                 *result,
                 &(*pG),
                 rctlsb,
                 hfembit,
                 *hbheDigis,
                 *hfDigis);
  } else if (legacy_ and upgrade_) {
    theAlgo_.run(inputCoder.product(),
                 outTranscoder->getHcalCompressor().get(),
                 pSetup.product(),
                 *result,
                 &(*pG),
                 rctlsb,
                 hfembit,
                 *hbheDigis,
                 *hfDigis,
                 *hbheUpDigis,
                 *hfUpDigis);
  } else {
    theAlgo_.run(inputCoder.product(),
                 outTranscoder->getHcalCompressor().get(),
                 pSetup.product(),
                 *result,
                 &(*pG),
                 rctlsb,
                 hfembit,
                 *hbheUpDigis,
                 *hfUpDigis);
  }

  // Step C.1: Run FE Format Error / ZS for real data.
  if (runFrontEndFormatError_) {
    const HcalElectronicsMap* emap = pSetup->getHcalMapping();

    edm::Handle<FEDRawDataCollection> fedHandle;
    iEvent.getByToken(tok_raw_, fedHandle);

    if (fedHandle.isValid() && emap != nullptr) {
      theAlgo_.runFEFormatError(fedHandle.product(), emap, *result);
    } else {
      edm::LogInfo("HcalTrigPrimDigiProducer") << "\nWarning: FEDRawDataCollection with input tag " << inputTagFEDRaw_
                                               << "\nrequested in configuration, but not found in the event."
                                               << "\nQuit returning empty product." << std::endl;

      // produce empty HcalTrigPrimDigiCollection and put it in the event
      std::unique_ptr<HcalTrigPrimDigiCollection> emptyResult(new HcalTrigPrimDigiCollection());

      iEvent.put(std::move(emptyResult));

      return;
    }
  }

  if (runZS_)
    theAlgo_.runZS(*result);

  //  edm::LogInfo("HcalTrigPrimDigiProducer") << "HcalTrigPrims: " << result->size();

  // Step D: Put outputs into event
  iEvent.put(std::move(result));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalTrigPrimDigiProducer);
