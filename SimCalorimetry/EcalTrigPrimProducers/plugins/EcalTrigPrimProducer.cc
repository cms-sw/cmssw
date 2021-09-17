
/** \class EcalTrigPrimProducer
 *
 * EcalTrigPrimProducer produces a EcalTrigPrimDigiCollection
 * The barrel code does a detailed simulation
 * The code for the endcap is simulated in a rough way, due to missing strip
 geometry
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni, LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006

 *
 ************************************************************/
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSlidingWindowRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGOddWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGOddWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGTPModeRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGOddWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGOddWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTPMode.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <memory>

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"

class EcalTrigPrimProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalTrigPrimProducer(const edm::ParameterSet &conf);

  ~EcalTrigPrimProducer() override;

  void beginRun(const edm::Run &run, const edm::EventSetup &es) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override;
  void produce(edm::Event &e, const edm::EventSetup &c) override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  std::unique_ptr<EcalTrigPrimFunctionalAlgo> algo_;
  bool barrelOnly_;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  bool tpInfoPrintout_;
  edm::EDGetTokenT<EBDigiCollection> tokenEB_;
  edm::EDGetTokenT<EEDigiCollection> tokenEE_;

  const edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> tokenEcalMapping_;
  //these are only used if we also handle the endcap
  edm::ESGetToken<CaloSubdetectorGeometry, EcalEndcapGeometryRecord> tokenEndcapGeom_;
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> tokenETTMap_;

  // for EcalFenixStrip...
  // get parameter records for xtals
  edm::ESGetToken<EcalTPGLinearizationConst, EcalTPGLinearizationConstRcd> tokenEcalTPGLinearization_;
  edm::ESGetToken<EcalTPGPedestals, EcalTPGPedestalsRcd> tokenEcalTPGPedestals_;
  edm::ESGetToken<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd> tokenEcalTPGCrystalStatus_;

  // for strips
  edm::ESGetToken<EcalTPGSlidingWindow, EcalTPGSlidingWindowRcd> tokenEcalTPGSlidingWindow_;
  edm::ESGetToken<EcalTPGWeightIdMap, EcalTPGWeightIdMapRcd> tokenEcalTPGWeightIdMap_;
  edm::ESGetToken<EcalTPGWeightGroup, EcalTPGWeightGroupRcd> tokenEcalTPGWeightGroup_;
  edm::ESGetToken<EcalTPGOddWeightIdMap, EcalTPGOddWeightIdMapRcd> tokenEcalTPGOddWeightIdMap_;
  edm::ESGetToken<EcalTPGOddWeightGroup, EcalTPGOddWeightGroupRcd> tokenEcalTPGOddWeightGroup_;
  edm::ESGetToken<EcalTPGFineGrainStripEE, EcalTPGFineGrainStripEERcd> tokenEcalTPGFineGrainStripEE_;
  edm::ESGetToken<EcalTPGStripStatus, EcalTPGStripStatusRcd> tokenEcalTPGStripStatus_;

  // .. and for EcalFenixTcp
  // get parameter records for towers
  edm::ESGetToken<EcalTPGFineGrainEBGroup, EcalTPGFineGrainEBGroupRcd> tokenEcalTPGFineGrainEBGroup_;
  edm::ESGetToken<EcalTPGLutGroup, EcalTPGLutGroupRcd> tokenEcalTPGLutGroup_;
  edm::ESGetToken<EcalTPGLutIdMap, EcalTPGLutIdMapRcd> tokenEcalTPGLutIdMap_;
  edm::ESGetToken<EcalTPGFineGrainEBIdMap, EcalTPGFineGrainEBIdMapRcd> tokenEcalTPGFineGrainEBIdMap_;
  edm::ESGetToken<EcalTPGFineGrainTowerEE, EcalTPGFineGrainTowerEERcd> tokenEcalTPGFineGrainTowerEE_;
  edm::ESGetToken<EcalTPGTowerStatus, EcalTPGTowerStatusRcd> tokenEcalTPGTowerStatus_;
  edm::ESGetToken<EcalTPGSpike, EcalTPGSpikeRcd> tokenEcalTPGSpike_;
  // TPG TP mode
  edm::ESGetToken<EcalTPGTPMode, EcalTPGTPModeRcd> tokenEcalTPGTPMode_;

  int binOfMaximum_;
  bool fillBinOfMaximumFromHistory_;

  // method to get EventSetupRecords
  unsigned long long getRecords(edm::EventSetup const &setup);
  unsigned long long cacheID_;
};

EcalTrigPrimProducer::EcalTrigPrimProducer(const edm::ParameterSet &iConfig)
    : barrelOnly_(iConfig.getParameter<bool>("BarrelOnly")),
      tcpFormat_(iConfig.getParameter<bool>("TcpOutput")),
      debug_(iConfig.getParameter<bool>("Debug")),
      famos_(iConfig.getParameter<bool>("Famos")),
      tpInfoPrintout_(iConfig.getParameter<bool>("TPinfoPrintout")),
      tokenEB_(consumes<EBDigiCollection>(
          edm::InputTag(iConfig.getParameter<std::string>("Label"), iConfig.getParameter<std::string>("InstanceEB")))),
      tokenEE_(consumes<EEDigiCollection>(
          edm::InputTag(iConfig.getParameter<std::string>("Label"), iConfig.getParameter<std::string>("InstanceEE")))),
      tokenEcalMapping_(esConsumes<edm::Transition::BeginRun>()),
      tokenEcalTPGLinearization_(esConsumes()),
      tokenEcalTPGPedestals_(esConsumes()),
      tokenEcalTPGCrystalStatus_(esConsumes()),
      tokenEcalTPGSlidingWindow_(esConsumes()),
      tokenEcalTPGWeightIdMap_(esConsumes()),
      tokenEcalTPGWeightGroup_(esConsumes()),
      tokenEcalTPGOddWeightIdMap_(esConsumes()),
      tokenEcalTPGOddWeightGroup_(esConsumes()),
      tokenEcalTPGFineGrainStripEE_(esConsumes()),
      tokenEcalTPGStripStatus_(esConsumes()),
      tokenEcalTPGFineGrainEBGroup_(esConsumes()),
      tokenEcalTPGLutGroup_(esConsumes()),
      tokenEcalTPGLutIdMap_(esConsumes()),
      tokenEcalTPGFineGrainEBIdMap_(esConsumes()),
      tokenEcalTPGFineGrainTowerEE_(esConsumes()),
      tokenEcalTPGTowerStatus_(esConsumes()),
      tokenEcalTPGSpike_(esConsumes()),
      tokenEcalTPGTPMode_(esConsumes()),
      binOfMaximum_(iConfig.getParameter<int>("binOfMaximum")),
      fillBinOfMaximumFromHistory_(-1 == binOfMaximum_),
      cacheID_(0) {
  // register your products
  produces<EcalTrigPrimDigiCollection>();
  if (tcpFormat_)
    produces<EcalTrigPrimDigiCollection>("formatTCP");
  if (not barrelOnly_) {
    tokenEndcapGeom_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "EcalEndcap"));
    tokenETTMap_ = esConsumes<edm::Transition::BeginRun>();
  }
}

static int findBinOfMaximum(bool iFillFromHistory, int iPSetValue, edm::ProcessHistory const &iHistory) {
  //  get  binOfMax
  //  try first in cfg, then in ProcessHistory
  //  =6 is default (1-10 possible values)
  int binOfMaximum = 0;  // starts at 1!
  if (not iFillFromHistory) {
    binOfMaximum = iPSetValue;
    edm::LogInfo("EcalTPG") << "EcalTrigPrimProducer is using binOfMaximum found in cfg file :  " << binOfMaximum;
  }

  // search backwards in history looking for the particular module
  const std::string kModuleName{"ecalUnsuppressedDigis"};
  for (auto it = iHistory.rbegin(), itEnd = iHistory.rend(); it != itEnd; ++it) {
    auto const &topLevelPSet = getParameterSet(it->parameterSetID());
    if (topLevelPSet.exists(kModuleName)) {
      int psetBinOfMax = topLevelPSet.getParameter<edm::ParameterSet>(kModuleName).getParameter<int>("binOfMaximum");

      if (not iFillFromHistory) {
        if (psetBinOfMax != binOfMaximum)
          edm::LogWarning("EcalTPG") << "binofMaximum given in configuration (=" << binOfMaximum
                                     << ") is different from the one found in ProductRegistration(=" << psetBinOfMax
                                     << ")!!!";
      } else {
        binOfMaximum = psetBinOfMax;
        edm::LogInfo("EcalTPG") << "EcalTrigPrimProducer is using binOfMaximum "
                                   "found in ProductRegistry :  "
                                << binOfMaximum;
      }
      break;
    }
  }
  if (binOfMaximum == 0) {
    binOfMaximum = 6;
    edm::LogWarning("EcalTPG") << "Could not find product registry of EBDigiCollection (label "
                                  "ecalUnsuppressedDigis), had to set the following parameters by "
                                  "Hand:  binOfMaximum="
                               << binOfMaximum;
  }
  return binOfMaximum;
}

void EcalTrigPrimProducer::beginRun(edm::Run const &run, edm::EventSetup const &setup) {
  // ProcessHistory is guaranteed to be constant for an entire Run
  binOfMaximum_ = findBinOfMaximum(fillBinOfMaximumFromHistory_, binOfMaximum_, run.processHistory());

  auto const &ecalmapping = setup.getData(tokenEcalMapping_);
  if (barrelOnly_) {
    algo_ = std::make_unique<EcalTrigPrimFunctionalAlgo>(
        &ecalmapping, binOfMaximum_, tcpFormat_, debug_, famos_, tpInfoPrintout_);
  } else {
    auto const &endcapGeometry = setup.getData(tokenEndcapGeom_);
    auto const &eTTmap = setup.getData(tokenETTMap_);
    algo_ = std::make_unique<EcalTrigPrimFunctionalAlgo>(
        &eTTmap, &endcapGeometry, &ecalmapping, binOfMaximum_, tcpFormat_, debug_, famos_, tpInfoPrintout_);
  }
}

void EcalTrigPrimProducer::endRun(edm::Run const &run, edm::EventSetup const &setup) {
  algo_.reset();
  cacheID_ = 0;
}

unsigned long long EcalTrigPrimProducer::getRecords(edm::EventSetup const &setup) {
  // get Eventsetup records

  // for EcalFenixStrip...
  // get parameter records for xtals
  const EcalTPGLinearizationConst *ecaltpLin = &setup.getData(tokenEcalTPGLinearization_);
  const EcalTPGPedestals *ecaltpPed = &setup.getData(tokenEcalTPGPedestals_);
  const EcalTPGCrystalStatus *ecaltpgBadX = &setup.getData(tokenEcalTPGCrystalStatus_);

  // for strips
  const EcalTPGSlidingWindow *ecaltpgSlidW = &setup.getData(tokenEcalTPGSlidingWindow_);
  const EcalTPGWeightIdMap *ecaltpgWeightMap = &setup.getData(tokenEcalTPGWeightIdMap_);
  const EcalTPGWeightGroup *ecaltpgWeightGroup = &setup.getData(tokenEcalTPGWeightGroup_);
  const EcalTPGOddWeightIdMap *ecaltpgOddWeightMap = &setup.getData(tokenEcalTPGOddWeightIdMap_);
  const EcalTPGOddWeightGroup *ecaltpgOddWeightGroup = &setup.getData(tokenEcalTPGOddWeightGroup_);
  const EcalTPGFineGrainStripEE *ecaltpgFgStripEE = &setup.getData(tokenEcalTPGFineGrainStripEE_);
  const EcalTPGStripStatus *ecaltpgStripStatus = &setup.getData(tokenEcalTPGStripStatus_);
  const EcalTPGTPMode *ecaltpgTPMode = &setup.getData(tokenEcalTPGTPMode_);

  algo_->setPointers(ecaltpLin,
                     ecaltpPed,
                     ecaltpgSlidW,
                     ecaltpgWeightMap,
                     ecaltpgWeightGroup,
                     ecaltpgOddWeightMap,
                     ecaltpgOddWeightGroup,
                     ecaltpgFgStripEE,
                     ecaltpgBadX,
                     ecaltpgStripStatus,
                     ecaltpgTPMode);

  // .. and for EcalFenixTcp
  // get parameter records for towers
  const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup = &setup.getData(tokenEcalTPGFineGrainEBGroup_);
  const EcalTPGLutGroup *ecaltpgLutGroup = &setup.getData(tokenEcalTPGLutGroup_);
  const EcalTPGLutIdMap *ecaltpgLut = &setup.getData(tokenEcalTPGLutIdMap_);
  const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB = &setup.getData(tokenEcalTPGFineGrainEBIdMap_);
  const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE = &setup.getData(tokenEcalTPGFineGrainTowerEE_);
  const EcalTPGTowerStatus *ecaltpgBadTT = &setup.getData(tokenEcalTPGTowerStatus_);
  const EcalTPGSpike *ecaltpgSpike = &setup.getData(tokenEcalTPGSpike_);

  algo_->setPointers2(ecaltpgFgEBGroup,
                      ecaltpgLutGroup,
                      ecaltpgLut,
                      ecaltpgFineGrainEB,
                      ecaltpgFineGrainTowerEE,
                      ecaltpgBadTT,
                      ecaltpgSpike,
                      ecaltpgTPMode);

  // we will suppose that everything is to be updated if the
  // EcalTPGLinearizationConstRcd has changed
  return setup.get<EcalTPGLinearizationConstRcd>().cacheIdentifier();
}

EcalTrigPrimProducer::~EcalTrigPrimProducer() {}

// ------------ method called to produce the data  ------------
void EcalTrigPrimProducer::produce(edm::Event &e, const edm::EventSetup &iSetup) {
  // update constants if necessary
  if (iSetup.get<EcalTPGLinearizationConstRcd>().cacheIdentifier() != cacheID_)
    cacheID_ = this->getRecords(iSetup);

  // get input collections

  edm::Handle<EBDigiCollection> ebDigis;
  edm::Handle<EEDigiCollection> eeDigis;
  bool barrel = true;
  bool endcap = true;
  if (barrelOnly_)
    endcap = false;

  if (!e.getByToken(tokenEB_, ebDigis)) {
    barrel = false;
    edm::EDConsumerBase::Labels labels;
    labelsForToken(tokenEB_, labels);
    edm::LogWarning("EcalTPG") << " Couldnt find Barrel dataframes with producer " << labels.module << " and label "
                               << labels.productInstance << "!!!";
  }
  if (!barrelOnly_) {
    if (!e.getByToken(tokenEE_, eeDigis)) {
      endcap = false;
      edm::EDConsumerBase::Labels labels;
      labelsForToken(tokenEE_, labels);
      edm::LogWarning("EcalTPG") << " Couldnt find Endcap dataframes with producer " << labels.module << " and label "
                                 << labels.productInstance << "!!!";
    }
  }
  if (!barrel && !endcap) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(tokenEB_, labels);
    throw cms::Exception(" ProductNotFound") << "No EBDataFrames(EEDataFrames) with producer " << labels.module
                                             << " and label " << labels.productInstance << " found in input!!\n";
  }

  if (!barrelOnly_)
    LogDebug("EcalTPG") << " =================> Treating event  " << e.id() << ", Number of EBDataFrames "
                        << ebDigis.product()->size() << ", Number of EEDataFrames " << eeDigis.product()->size();
  else
    LogDebug("EcalTPG") << " =================> Treating event  " << e.id() << ", Number of EBDataFrames "
                        << ebDigis.product()->size();

  auto pOut = std::make_unique<EcalTrigPrimDigiCollection>();
  auto pOutTcp = std::make_unique<EcalTrigPrimDigiCollection>();

  // invoke algorithm

  const EBDigiCollection *ebdc = nullptr;
  const EEDigiCollection *eedc = nullptr;
  if (barrel) {
    ebdc = ebDigis.product();
    algo_->run(ebdc, *pOut, *pOutTcp);
  }

  if (endcap) {
    eedc = eeDigis.product();
    algo_->run(eedc, *pOut, *pOutTcp);
  }

  edm::LogInfo("produce") << "For Barrel + Endcap, " << pOut->size() << " TP  Digis were produced";

  //  debug prints if TP >0

  for (unsigned int i = 0; i < pOut->size(); ++i) {
    bool print = false;
    for (int isam = 0; isam < (*pOut)[i].size(); ++isam) {
      if ((*pOut)[i][isam].raw())
        print = true;
    }
    if (print)
      LogDebug("EcalTPG") << " For tower  " << (((*pOut)[i])).id() << ", TP is " << (*pOut)[i];
  }
  if (barrelOnly_)
    LogDebug("EcalTPG") << "\n =================> For Barrel , " << pOut->size()
                        << " TP  Digis were produced (including zero ones)";
  else
    LogDebug("EcalTPG") << "\n =================> For Barrel + Endcap, " << pOut->size()
                        << " TP  Digis were produced (including zero ones)";

  // put result into the Event

  e.put(std::move(pOut));
  if (tcpFormat_)
    e.put(std::move(pOutTcp), "formatTCP");
}

void EcalTrigPrimProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("BarrelOnly", false);
  desc.add<bool>("TcpOutput", false);
  desc.add<bool>("Debug", false);
  desc.add<bool>("Famos", false);
  desc.add<std::string>("Label", "simEcalUnsuppressedDigis");
  desc.add<std::string>("InstanceEB", "");
  desc.add<std::string>("InstanceEE", "");
  const std::string kComment(
      "A value of -1 will make the module lookup the value of 'binOfMaximum' "
      "from the module 'ecalUnsuppressedDigis' from the process history. "
      "Allowed values are -1 and from 1-10.");
  // The code before the existence of fillDescriptions did something special if
  // 'binOfMaximum' was missing. This replicates that behavior.
  desc.add<int>("binOfMaximum", -1)->setComment(kComment);
  desc.add<bool>("TPinfoPrintout", false);
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(EcalTrigPrimProducer);
