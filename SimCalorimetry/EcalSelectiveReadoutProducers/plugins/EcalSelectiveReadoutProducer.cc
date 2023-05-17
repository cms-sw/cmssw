#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/interface/EcalSelectiveReadoutSuppressor.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include <memory>
#include <vector>

#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"

#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/namespace_ecalsrcondtools.h"

#include <memory>
#include <fstream>
#include <atomic>

namespace esrp {
  /* All stream instances of the module write to the same files. The writes are guarded
     by the shared mutex.
   */
  struct Cache {
    CMS_THREAD_GUARD("mutex_") mutable int iEvent_ = 1;
    mutable std::mutex mutex_;
  };
}  // namespace esrp

class EcalSelectiveReadoutProducer : public edm::stream::EDProducer<edm::GlobalCache<esrp::Cache>> {
public:
  /** Constructor
   * @param params seletive readout parameters
   */
  explicit EcalSelectiveReadoutProducer(const edm::ParameterSet& params, esrp::Cache const*);

  /** Destructor
   */

  ~EcalSelectiveReadoutProducer() override;

  /** Produces the EDM products
   * @param CMS event
   * @param eventSetup event conditions
   */
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

  /** Help function to print SR flags.
   * @param ebSrFlags the action flags of EB
   * @param eeSrFlag the action flags of EE
   * @param iEvent event number. Ignored if <0.
   * @param withHeader, if true an output description is written out as header.
   */
  static void printSrFlags(std::ostream& os,
                           const EBSrFlagCollection& ebSrFlags,
                           const EESrFlagCollection& eeSrFlags,
                           int iEvent = -1,
                           bool withHeader = true);

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc);

  static std::unique_ptr<esrp::Cache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(esrp::Cache*) {}

private:
  /** Sanity check on the DCC FIR filter weights. Log warning or
   * error message if an unexpected weight set is found. In principle
   * it is checked that the maximum weight is applied to the expected
   * maximum sample.
   */
  void checkWeights(const edm::Event& evt, const edm::ProductID& noZSDigiId) const;

  /** Gets the value of the digitizer binOfMaximum parameter.
   * @param noZsDigiId product ID of the non-suppressed digis
   * @param binOfMax [out] set the parameter value if found
   * @return true on success, false otherwise
   */
  bool getBinOfMax(const edm::Event& evt, const edm::ProductID& noZsDigiId, int& binOfMax) const;

  const EBDigiCollection* getEBDigis(edm::Event& event);

  const EEDigiCollection* getEEDigis(edm::Event& event);

  const EcalTrigPrimDigiCollection* getTrigPrims(edm::Event& event) const;

  ///@{
  /// call these once an event, to make sure everything
  /// is up-to-date
  void checkGeometry(const edm::EventSetup& eventSetup);
  void checkTriggerMap(const edm::EventSetup& eventSetup);
  void checkElecMap(const edm::EventSetup& eventSetup);

  ///@}

  ///Checks validity of selective setting object is valid to be used
  ///for MC, especially checks the number of elements in the vectors
  ///@param forEmulator if true check the restriction that applies for
  ///EcalSelectiveReadoutProducer
  ///@throw cms::Exception if the setting is not valid.
  static void checkValidity(const EcalSRSettings& settings);

  void printTTFlags(const EcalTrigPrimDigiCollection& tp, std::ostream& os) const;

private:
  EcalSelectiveReadoutSuppressor suppressor_;
  std::string digiProducer_;         // name of module/plugin/producer making digis
  std::string ebdigiCollection_;     // secondary name given to collection of input digis
  std::string eedigiCollection_;     // secondary name given to collection of input digis
  std::string ebSRPdigiCollection_;  // secondary name given to collection of suppressed digis
  std::string eeSRPdigiCollection_;  // secondary name given to collection of suppressed digis
  std::string ebSrFlagCollection_;   // secondary name given to collection of SR flag digis
  std::string eeSrFlagCollection_;   // secondary name given to collection of SR flag digis
  std::string trigPrimProducer_;     // name of module/plugin/producer making triggere primitives
  std::string trigPrimCollection_;   // name of module/plugin/producer making triggere primitives

  // store the pointer, so we don't have to update it every event
  const CaloGeometry* theGeometry;
  const EcalTrigTowerConstituentsMap* theTriggerTowerMap;
  const EcalElectronicsMapping* theElecMap;

  bool suppressorSettingsSet_ = false;

  bool trigPrimBypass_;

  int trigPrimBypassMode_;

  /** Number of event whose TT and SR flags must be dumped into a file.
   */
  int dumpFlags_;

  /** switch to write out the SrFlags collections in the event
   */
  bool writeSrFlags_;

  /** Switch for suppressed digi production If false SR flags are produced
   * but selective readout is not applied on the crystal channel digis.
   */
  bool produceDigis_;

  /** SR settings
   */
  const EcalSRSettings* settings_;

  /** Switch for retrieving SR settings from condition database instead
   * of CMSSW python configuration file.
   */
  bool useCondDb_;

  /**  Special switch to turn off SR entirely using special DB entries 
   */

  bool useFullReadout_;

  /** keys
   */
  bool firstCallEB_;
  bool firstCallEE_;

  /** Used when settings_ is imported from configuration file. Just used
   * for memory management. Used settings_ to access to the object
   */
  std::unique_ptr<EcalSRSettings> settingsFromFile_;

  // Tokens for consumes collection:

  edm::EDGetTokenT<EBDigiCollection> EB_token;
  edm::EDGetTokenT<EEDigiCollection> EE_token;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> EcTP_token;
  edm::ESGetToken<EcalSRSettings, EcalSRSettingsRcd> hSr_token_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geom_token_;
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> eTTmap_token_;
  edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> eElecmap_token_;
};

using namespace std;
using namespace ecalsrcondtools;

std::unique_ptr<esrp::Cache> EcalSelectiveReadoutProducer::initializeGlobalCache(const edm::ParameterSet&) {
  return std::make_unique<esrp::Cache>();
}

EcalSelectiveReadoutProducer::EcalSelectiveReadoutProducer(const edm::ParameterSet& params, esrp::Cache const*)
    : suppressor_(params, consumesCollector()), firstCallEB_(true), firstCallEE_(true) {
  //settings:
  //  settings which are only in python config files:
  digiProducer_ = params.getParameter<string>("digiProducer");
  ebdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
  eedigiCollection_ = params.getParameter<std::string>("EEdigiCollection");
  ebSRPdigiCollection_ = params.getParameter<std::string>("EBSRPdigiCollection");
  eeSRPdigiCollection_ = params.getParameter<std::string>("EESRPdigiCollection");
  ebSrFlagCollection_ = params.getParameter<std::string>("EBSrFlagCollection");
  eeSrFlagCollection_ = params.getParameter<std::string>("EESrFlagCollection");
  trigPrimProducer_ = params.getParameter<string>("trigPrimProducer");
  trigPrimCollection_ = params.getParameter<string>("trigPrimCollection");
  trigPrimBypass_ = params.getParameter<bool>("trigPrimBypass");
  trigPrimBypassMode_ = params.getParameter<int>("trigPrimBypassMode");
  dumpFlags_ = params.getUntrackedParameter<int>("dumpFlags");
  writeSrFlags_ = params.getUntrackedParameter<bool>("writeSrFlags");
  produceDigis_ = params.getUntrackedParameter<bool>("produceDigis");
  //   settings which can come from either condition database or python configuration file:
  useCondDb_ = params.getParameter<bool>("configFromCondDB");
  if (!useCondDb_) {
    settingsFromFile_ = std::make_unique<EcalSRSettings>();
    ecalsrcondtools::importParameterSet(*settingsFromFile_, params);
    settings_ = settingsFromFile_.get();
  }

  //declares the products made by this producer:
  if (produceDigis_) {
    produces<EBDigiCollection>(ebSRPdigiCollection_);
    produces<EEDigiCollection>(eeSRPdigiCollection_);
  }

  if (writeSrFlags_) {
    produces<EBSrFlagCollection>(ebSrFlagCollection_);
    produces<EESrFlagCollection>(eeSrFlagCollection_);
  }

  useFullReadout_ = params.getParameter<bool>("UseFullReadout");

  theGeometry = nullptr;
  theTriggerTowerMap = nullptr;
  theElecMap = nullptr;

  EB_token = consumes<EBDigiCollection>(edm::InputTag(digiProducer_, ebdigiCollection_));
  EE_token = consumes<EEDigiCollection>(edm::InputTag(digiProducer_, eedigiCollection_));
  ;
  EcTP_token = consumes<EcalTrigPrimDigiCollection>(edm::InputTag(trigPrimProducer_, trigPrimCollection_));
  if (useFullReadout_) {
    hSr_token_ = esConsumes<EcalSRSettings, EcalSRSettingsRcd>(edm::ESInputTag("", "fullReadout"));
  } else {
    hSr_token_ = esConsumes<EcalSRSettings, EcalSRSettingsRcd>();
  }
  geom_token_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  eTTmap_token_ = esConsumes<EcalTrigTowerConstituentsMap, IdealGeometryRecord>();
  eElecmap_token_ = esConsumes<EcalElectronicsMapping, EcalMappingRcd>();
  ;
}

void EcalSelectiveReadoutProducer::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
  edm::ParameterSetDescription ps;

  ps.add<string>("digiProducer");
  ps.add<std::string>("EBdigiCollection");
  ps.add<std::string>("EEdigiCollection");
  ps.add<std::string>("EBSRPdigiCollection");
  ps.add<std::string>("EESRPdigiCollection");
  ps.add<std::string>("EBSrFlagCollection");
  ps.add<std::string>("EESrFlagCollection");
  ps.add<string>("trigPrimProducer");
  ps.add<string>("trigPrimCollection");
  ps.add<bool>("trigPrimBypass");     //also used by suppressor
  ps.add<int>("trigPrimBypassMode");  // also used by suppressor
  ps.addUntracked<int>("dumpFlags", 0);
  ps.addUntracked<bool>("writeSrFlags", false);
  ps.addUntracked<bool>("produceDigis", true);
  ps.add<bool>("configFromCondDB", false);
  ps.add<bool>("UseFullReadout");

  //from suppressor_
  ps.add<int>("defaultTtf");
  ps.add<bool>("trigPrimBypassWithPeakFinder");
  ps.add<double>("trigPrimBypassLTH");
  ps.add<double>("trigPrimBypassHTH");

  //from importParameterSet
  ps.addOptional<int>("deltaPhi");
  ps.addOptional<int>("deltaEta");
  ps.addOptional<int>("ecalDccZs1stSample");
  ps.addOptional<double>("ebDccAdcToGeV");
  ps.addOptional<double>("eeDccAdcToGeV");
  ps.addOptional<std::vector<double>>("dccNormalizedWeights");
  ps.addOptional<bool>("symetricZS");
  ps.addOptional<double>("srpBarrelLowInterestChannelZS");
  ps.addOptional<double>("srpEndcapLowInterestChannelZS");
  ps.addOptional<double>("srpBarrelHighInterestChannelZS");
  ps.addOptional<double>("srpEndcapHighInterestChannelZS");

  iDesc.addDefault(ps);
}

EcalSelectiveReadoutProducer::~EcalSelectiveReadoutProducer() {}

void EcalSelectiveReadoutProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  if (useCondDb_) {
    //getting selective readout configuration:
    edm::ESHandle<EcalSRSettings> hSr = eventSetup.getHandle(hSr_token_);

    settings_ = hSr.product();
  }

  //gets the trigger primitives:
  EcalTrigPrimDigiCollection emptyTPColl;
  const EcalTrigPrimDigiCollection* trigPrims =
      (trigPrimBypass_ && trigPrimBypassMode_ == 0) ? &emptyTPColl : getTrigPrims(event);

  //gets the digis from the events:
  EBDigiCollection dummyEbDigiColl;
  EEDigiCollection dummyEeDigiColl;

  const EBDigiCollection* ebDigis = produceDigis_ ? getEBDigis(event) : &dummyEbDigiColl;
  const EEDigiCollection* eeDigis = produceDigis_ ? getEEDigis(event) : &dummyEeDigiColl;

  //runs the selective readout algorithm:
  unique_ptr<EBDigiCollection> selectedEBDigis;
  unique_ptr<EEDigiCollection> selectedEEDigis;
  unique_ptr<EBSrFlagCollection> ebSrFlags;
  unique_ptr<EESrFlagCollection> eeSrFlags;

  if (produceDigis_) {
    selectedEBDigis = std::make_unique<EBDigiCollection>();
    selectedEEDigis = std::make_unique<EEDigiCollection>();
  }

  if (writeSrFlags_) {
    ebSrFlags = std::make_unique<EBSrFlagCollection>();
    eeSrFlags = std::make_unique<EESrFlagCollection>();
  }

  if (not suppressorSettingsSet_) {
    //Check the validity of EcalSRSettings
    checkValidity(*settings_);

    suppressor_.setSettings(settings_);
    suppressorSettingsSet_ = true;

    // check that everything is up-to-date
    checkGeometry(eventSetup);
    checkTriggerMap(eventSetup);
    checkElecMap(eventSetup);
  }

  suppressor_.run(eventSetup,
                  *trigPrims,
                  *ebDigis,
                  *eeDigis,
                  selectedEBDigis.get(),
                  selectedEEDigis.get(),
                  ebSrFlags.get(),
                  eeSrFlags.get());

  if (dumpFlags_ > 0) {
    auto* cache = globalCache();
    std::lock_guard<std::mutex> guard(cache->mutex_);
    auto iEvent_ = cache->iEvent_;
    if (dumpFlags_ >= iEvent_) {
      ofstream ttfFile("TTF.txt", (iEvent_ == 1 ? ios::trunc : ios::app));
      suppressor_.printTTFlags(ttfFile, iEvent_, iEvent_ == 1 ? true : false);

      ofstream srfFile("SRF.txt", (iEvent_ == 1 ? ios::trunc : ios::app));
      if (iEvent_ == 1) {
        suppressor_.getEcalSelectiveReadout()->printHeader(srfFile);
      }
      srfFile << "# Event " << iEvent_ << "\n";
      suppressor_.getEcalSelectiveReadout()->print(srfFile);
      srfFile << "\n";

      ofstream afFile("AF.txt", (iEvent_ == 1 ? ios::trunc : ios::app));
      printSrFlags(afFile, *ebSrFlags, *eeSrFlags, iEvent_, iEvent_ == 1 ? true : false);
      ++(cache->iEvent_);
    } else {
      //do not want to dump anymore, so can turn off
      dumpFlags_ = 0;
    }
  }

  if (produceDigis_) {
    //puts the selected digis into the event:
    event.put(std::move(selectedEBDigis), ebSRPdigiCollection_);
    event.put(std::move(selectedEEDigis), eeSRPdigiCollection_);
  }

  //puts the SR flags into the event:
  if (writeSrFlags_) {
    event.put(std::move(ebSrFlags), ebSrFlagCollection_);
    event.put(std::move(eeSrFlags), eeSrFlagCollection_);
  }
}

const EBDigiCollection* EcalSelectiveReadoutProducer::getEBDigis(edm::Event& event) {
  edm::Handle<EBDigiCollection> hEBDigis;
  event.getByToken(EB_token, hEBDigis);
  //product() method is called before id() in order to get an exception
  //if the handle is not available (check not done by id() method).
  const EBDigiCollection* result = hEBDigis.product();
  if (firstCallEB_) {
    checkWeights(event, hEBDigis.id());
    firstCallEB_ = false;
  }
  return result;
}

const EEDigiCollection* EcalSelectiveReadoutProducer::getEEDigis(edm::Event& event) {
  edm::Handle<EEDigiCollection> hEEDigis;
  event.getByToken(EE_token, hEEDigis);
  //product() method is called before id() in order to get an exception
  //if the handle is not available (check not done by id() method).
  const EEDigiCollection* result = hEEDigis.product();
  if (firstCallEE_) {
    checkWeights(event, hEEDigis.id());
    firstCallEE_ = false;
  }
  return result;
}

const EcalTrigPrimDigiCollection* EcalSelectiveReadoutProducer::getTrigPrims(edm::Event& event) const {
  edm::Handle<EcalTrigPrimDigiCollection> hTPDigis;
  event.getByToken(EcTP_token, hTPDigis);
  return hTPDigis.product();
}

void EcalSelectiveReadoutProducer::checkGeometry(const edm::EventSetup& eventSetup) {
  edm::ESHandle<CaloGeometry> hGeometry = eventSetup.getHandle(geom_token_);

  const CaloGeometry* pGeometry = &*hGeometry;

  // see if we need to update
  if (pGeometry != theGeometry) {
    theGeometry = pGeometry;
    suppressor_.setGeometry(theGeometry);
  }
}

void EcalSelectiveReadoutProducer::checkTriggerMap(const edm::EventSetup& eventSetup) {
  edm::ESHandle<EcalTrigTowerConstituentsMap> eTTmap = eventSetup.getHandle(eTTmap_token_);

  const EcalTrigTowerConstituentsMap* pMap = &*eTTmap;

  // see if we need to update
  if (pMap != theTriggerTowerMap) {
    theTriggerTowerMap = pMap;
    suppressor_.setTriggerMap(theTriggerTowerMap);
  }
}

void EcalSelectiveReadoutProducer::checkElecMap(const edm::EventSetup& eventSetup) {
  edm::ESHandle<EcalElectronicsMapping> eElecmap = eventSetup.getHandle(eElecmap_token_);

  const EcalElectronicsMapping* pMap = &*eElecmap;

  // see if we need to update
  if (pMap != theElecMap) {
    theElecMap = pMap;
    suppressor_.setElecMap(theElecMap);
  }
}

void EcalSelectiveReadoutProducer::printTTFlags(const EcalTrigPrimDigiCollection& tp, ostream& os) const {
  const char tccFlagMarker[] = {'x', '.', 'S', '?', 'C', 'E', 'E', 'E', 'E'};
  const int nEta = EcalSelectiveReadout::nTriggerTowersInEta;
  const int nPhi = EcalSelectiveReadout::nTriggerTowersInPhi;

  //static bool firstCall=true;
  //  if(firstCall){
  //  firstCall=false;
  os << "# TCC flag map\n#\n"
        "# +-->Phi            "
     << tccFlagMarker[0 + 1]
     << ": 000 (low interest)\n"
        "# |                  "
     << tccFlagMarker[1 + 1]
     << ": 001 (mid interest)\n"
        "# |                  "
     << tccFlagMarker[2 + 1]
     << ": 010 (not valid)\n"
        "# V Eta              "
     << tccFlagMarker[3 + 1]
     << ": 011 (high interest)\n"
        "#                    "
     << tccFlagMarker[4 + 1]
     << ": 1xx forced readout (Hw error)\n"
        "#\n";
  //}

  vector<vector<int>> ttf(nEta, vector<int>(nPhi, -1));
  for (EcalTrigPrimDigiCollection::const_iterator it = tp.begin(); it != tp.end(); ++it) {
    const EcalTriggerPrimitiveDigi& trigPrim = *it;
    if (trigPrim.size() > 0) {
      int iEta = trigPrim.id().ieta();
      int iEta0 = iEta < 0 ? iEta + nEta / 2 : iEta + nEta / 2 - 1;
      int iPhi0 = trigPrim.id().iphi() - 1;
      ttf[iEta0][iPhi0] = trigPrim.ttFlag();
    }
  }
  for (int iEta = 0; iEta < nEta; ++iEta) {
    for (int iPhi = 0; iPhi < nPhi; ++iPhi) {
      os << tccFlagMarker[ttf[iEta][iPhi] + 1];
    }
    os << "\n";
  }
}

void EcalSelectiveReadoutProducer::checkWeights(const edm::Event& evt, const edm::ProductID& noZsDigiId) const {
  const vector<float>& weights =
      settings_->dccNormalizedWeights_[0];  //params_.getParameter<vector<double> >("dccNormalizedWeights");
  int nFIRTaps = EcalSelectiveReadoutSuppressor::getFIRTapCount();
  static std::atomic<bool> warnWeightCnt{true};
  bool expected = true;
  if ((int)weights.size() > nFIRTaps &&
      warnWeightCnt.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
    edm::LogWarning("Configuration") << "The list of DCC zero suppression FIR "
                                        "weights given in parameter dccNormalizedWeights is longer "
                                        "than the expected depth of the FIR filter :("
                                     << nFIRTaps
                                     << "). "
                                        "The last weights will be discarded.";
  }

  if (!weights.empty()) {
    int iMaxWeight = 0;
    double maxWeight = weights[iMaxWeight];
    //looks for index of maximum weight
    for (unsigned i = 0; i < weights.size(); ++i) {
      if (weights[i] > maxWeight) {
        iMaxWeight = i;
        maxWeight = weights[iMaxWeight];
      }
    }

    //position of time sample whose maximum weight is applied:
    int maxWeightBin = settings_->ecalDccZs1stSample_[0]  //params_.getParameter<int>("ecalDccZs1stSample")
                       + iMaxWeight;

    //gets the bin of maximum (in case of raw data it will not exist)
    int binOfMax = 0;
    bool rc = getBinOfMax(evt, noZsDigiId, binOfMax);

    if (rc && maxWeightBin != binOfMax) {
      edm::LogWarning("Configuration") << "The maximum weight of DCC zero suppression FIR filter is not "
                                          "applied to the expected maximum sample("
                                       << binOfMax
                                       << (binOfMax == 1 ? "st"
                                                         : (binOfMax == 2 ? "nd" : (binOfMax == 3 ? "rd" : "th")))
                                       << " time sample). This may indicate faulty 'dccNormalizedWeights' "
                                          "or 'ecalDccZs1sSample' parameters.";
    }
  }
}

bool EcalSelectiveReadoutProducer::getBinOfMax(const edm::Event& evt,
                                               const edm::ProductID& noZsDigiId,
                                               int& binOfMax) const {
  bool rc;
  const edm::StableProvenance& p = evt.getStableProvenance(noZsDigiId);
  const edm::ParameterSet& result = parameterSet(p, evt.processHistory());
  vector<string> ebDigiParamList = result.getParameterNames();
  string bofm("binOfMaximum");
  if (find(ebDigiParamList.begin(), ebDigiParamList.end(), bofm) != ebDigiParamList.end()) {  //bofm found
    binOfMax = result.getParameter<int>("binOfMaximum");
    rc = true;
  } else {
    rc = false;
  }
  return rc;
}

void EcalSelectiveReadoutProducer::printSrFlags(ostream& os,
                                                const EBSrFlagCollection& ebSrFlags,
                                                const EESrFlagCollection& eeSrFlags,
                                                int iEvent,
                                                bool withHeader) {
  const char srpFlagMarker[] = {'.', 'z', 'Z', 'F', '4', '5', '6', '7'};
  if (withHeader) {
    time_t t;
    time(&t);
    const char* date = ctime(&t);
    os << "#SRP flag map\n#\n"
          "# Generatied on: "
       << date
       << "\n#\n"
          "# +-->Phi/Y "
       << srpFlagMarker[0]
       << ": suppressed\n"
          "# |         "
       << srpFlagMarker[1]
       << ": ZS 1\n"
          "# |         "
       << srpFlagMarker[2]
       << ": ZS 2\n"
          "# V Eta/X   "
       << srpFlagMarker[3]
       << ": full readout\n"
          "#\n";
  }

  //EE-,EB,EE+ map wil be written onto file in following format:
  //
  //      72
  // <-------------->
  //  20
  // <--->
  //  EEE                A             +-----> Y
  // EEEEE               |             |
  // EE EE               | 20   EE-    |
  // EEEEE               |             |
  //  EEE                V             V X
  // BBBBBBBBBBBBBBBBB   A
  // BBBBBBBBBBBBBBBBB   |             +-----> Phi
  // BBBBBBBBBBBBBBBBB   |             |
  // BBBBBBBBBBBBBBBBB   | 34  EB      |
  // BBBBBBBBBBBBBBBBB   |             |
  // BBBBBBBBBBBBBBBBB   |             V Eta
  // BBBBBBBBBBBBBBBBB   |
  // BBBBBBBBBBBBBBBBB   |
  // BBBBBBBBBBBBBBBBB   V
  //  EEE                A             +-----> Y
  // EEEEE               |             |
  // EE EE               | 20 EE+      |
  // EEEEE               |             |
  //  EEE                V             V X
  //
  //
  //
  //
  //event header:
  if (iEvent >= 0) {
    os << "# Event " << iEvent << "\n";
  }

  //retrieve flags:
  const int nEndcaps = 2;
  const int nScX = 20;
  const int nScY = 20;
  int eeSrf[nEndcaps][nScX][nScY];
  for (size_t i = 0; i < sizeof(eeSrf) / sizeof(int); ((int*)eeSrf)[i++] = -1) {
  };
  for (EESrFlagCollection::const_iterator it = eeSrFlags.begin(); it != eeSrFlags.end(); ++it) {
    const EESrFlag& flag = *it;
    int iZ0 = flag.id().zside() > 0 ? 1 : 0;
    int iX0 = flag.id().ix() - 1;
    int iY0 = flag.id().iy() - 1;
    assert(iZ0 >= 0 && iZ0 < nEndcaps);
    assert(iX0 >= 0 && iX0 < nScX);
    assert(iY0 >= 0 && iY0 < nScY);
    eeSrf[iZ0][iX0][iY0] = flag.value();
  }
  const int nEbTtEta = 34;
  const int nEeTtEta = 11;
  const int nTtEta = nEeTtEta * 2 + nEbTtEta;
  const int nTtPhi = 72;
  int ebSrf[nEbTtEta][nTtPhi];
  for (size_t i = 0; i < sizeof(ebSrf) / sizeof(int); ((int*)ebSrf)[i++] = -1) {
  };
  for (EBSrFlagCollection::const_iterator it = ebSrFlags.begin(); it != ebSrFlags.end(); ++it) {
    const EBSrFlag& flag = *it;
    int iEta = flag.id().ieta();
    int iEta0 = iEta + nTtEta / 2 - (iEta >= 0 ? 1 : 0);  //0->55 from eta=-3 to eta=3
    int iEbEta0 = iEta0 - nEeTtEta;                       //0->33 from eta=-1.48 to eta=1.48
    int iPhi0 = flag.id().iphi() - 1;
    assert(iEbEta0 >= 0 && iEbEta0 < nEbTtEta);
    assert(iPhi0 >= 0 && iPhi0 < nTtPhi);

    //     cout << __FILE__ << ":" << __LINE__ << ": "
    //	 <<  iEta << "\t" << flag.id().iphi() << " -> "
    //	 << iEbEta0 << "\t" << iPhi0
    //	 << "... Flag: " << flag.value() << "\n";

    ebSrf[iEbEta0][iPhi0] = flag.value();
  }

  //print flags:

  //EE-
  for (int iX0 = 0; iX0 < nScX; ++iX0) {
    for (int iY0 = 0; iY0 < nScY; ++iY0) {
      int srFlag = eeSrf[0][iX0][iY0];
      assert(srFlag >= -1 && srFlag < (int)(sizeof(srpFlagMarker) / sizeof(srpFlagMarker[0])));
      os << (srFlag == -1 ? ' ' : srpFlagMarker[srFlag]);
    }
    os << "\n";  //one Y supercystal column per line
  }              //next supercrystal X-index

  //EB
  for (int iEta0 = 0; iEta0 < nEbTtEta; ++iEta0) {
    for (int iPhi0 = 0; iPhi0 < nTtPhi; ++iPhi0) {
      int srFlag = ebSrf[iEta0][iPhi0];
      assert(srFlag >= -1 && srFlag < (int)(sizeof(srpFlagMarker) / sizeof(srpFlagMarker[0])));
      os << (srFlag == -1 ? '?' : srpFlagMarker[srFlag]);
    }
    os << "\n";  //one phi per line
  }

  //EE+
  for (int iX0 = 0; iX0 < nScX; ++iX0) {
    for (int iY0 = 0; iY0 < nScY; ++iY0) {
      int srFlag = eeSrf[1][iX0][iY0];
      assert(srFlag >= -1 && srFlag < (int)(sizeof(srpFlagMarker) / sizeof(srpFlagMarker[0])));
      os << (srFlag == -1 ? ' ' : srpFlagMarker[srFlag]);
    }
    os << "\n";  //one Y supercystal column per line
  }              //next supercrystal X-index

  //event trailer:
  os << "\n";
}

void EcalSelectiveReadoutProducer::checkValidity(const EcalSRSettings& settings) {
  if (settings.dccNormalizedWeights_.size() != 1) {
    throw cms::Exception("Configuration")
        << "Selective readout emulator, EcalSelectiveReadout, supports only single set of ZS weights. "
           "while the configuration contains "
        << settings.dccNormalizedWeights_.size() << " set(s)\n";
  }

  //   if(settings.dccNormalizedWeights_.size() != 1
  //      && settings.dccNormalizedWeights_.size() != 2
  //      && settings.dccNormalizedWeights_.size() != 54
  //      && settings.dccNormalizedWeights_.size() != 75848){
  //     throw cms::Exception("Configuration") << "Invalid number of DCC weight set (" << settings.dccNormalizedWeights_.size()
  //       << ") in condition object EcalSRSetting::dccNormalizedWeights_. "
  //       << "Valid counts are: 1 (single set), 2 (EB and EE), 54 (one per DCC) and 75848 "
  //       "(one per crystal)\n";
  //   }

  if (settings.dccNormalizedWeights_.size() != settings.ecalDccZs1stSample_.size()) {
    throw cms::Exception("Configuration")
        << "Inconsistency between number of weigth sets (" << settings.dccNormalizedWeights_.size() << ") and "
        << "number of ecalDccZs1Sample values (" << settings.ecalDccZs1stSample_.size() << ").";
  }
}

DEFINE_FWK_MODULE(EcalSelectiveReadoutProducer);
