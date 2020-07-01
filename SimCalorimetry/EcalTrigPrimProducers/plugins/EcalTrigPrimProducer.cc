
/** \class EcalTrigPrimProducer
 *
 * EcalTrigPrimProducer produces a EcalTrigPrimDigiCollection
 * The barrel code does a detailed simulation
 * The code for the endcap is simulated in a rough way, due to missing  strip
 geometry
 *
 *
 * \author Ursula Berthon, Stephanie Baffioni,  LLR Palaiseau
 *
 * \version   1st Version may 2006
 * \version   2nd Version jul 2006

 *
 ************************************************************/
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

#include "EcalTrigPrimProducer.h"


        #include <memory>

        #include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"

EcalTrigPrimProducer::EcalTrigPrimProducer(const edm::ParameterSet &iConfig)
    : barrelOnly_(iConfig.getParameter<bool>("BarrelOnly")),
      tcpFormat_(iConfig.getParameter<bool>("TcpOutput")),
      debug_(iConfig.getParameter<bool>("Debug")),
      famos_(iConfig.getParameter<bool>("Famos")),
      tokenEB_(consumes<EBDigiCollection>(
          edm::InputTag(iConfig.getParameter<std::string>("Label"), iConfig.getParameter<std::string>("InstanceEB")))),
      tokenEE_(consumes<EEDigiCollection>(
          edm::InputTag(iConfig.getParameter<std::string>("Label"), iConfig.getParameter<std::string>("InstanceEE")))),
      binOfMaximum_(iConfig.getParameter<int>("binOfMaximum")),
      fillBinOfMaximumFromHistory_(-1 == binOfMaximum_) {
  // register your products
  produces<EcalTrigPrimDigiCollection>();
  if (tcpFormat_)
    produces<EcalTrigPrimDigiCollection>("formatTCP");
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

  algo_ = std::make_unique<EcalTrigPrimFunctionalAlgo>(setup, binOfMaximum_, tcpFormat_, barrelOnly_, debug_, famos_);

  // get a first version of the records
  cacheID_ = this->getRecords(setup);
}

void EcalTrigPrimProducer::endRun(edm::Run const &run, edm::EventSetup const &setup) { algo_.reset(); }

unsigned long long EcalTrigPrimProducer::getRecords(edm::EventSetup const &setup) {
  // get Eventsetup records

  // for EcalFenixStrip...
  // get parameter records for xtals
  edm::ESHandle<EcalTPGLinearizationConst> theEcalTPGLinearization_handle;
  setup.get<EcalTPGLinearizationConstRcd>().get(theEcalTPGLinearization_handle);
  const EcalTPGLinearizationConst *ecaltpLin = theEcalTPGLinearization_handle.product();
  edm::ESHandle<EcalTPGPedestals> theEcalTPGPedestals_handle;
  setup.get<EcalTPGPedestalsRcd>().get(theEcalTPGPedestals_handle);
  const EcalTPGPedestals *ecaltpPed = theEcalTPGPedestals_handle.product();
  edm::ESHandle<EcalTPGCrystalStatus> theEcalTPGCrystalStatus_handle;
  setup.get<EcalTPGCrystalStatusRcd>().get(theEcalTPGCrystalStatus_handle);
  const EcalTPGCrystalStatus *ecaltpgBadX = theEcalTPGCrystalStatus_handle.product();

  // for strips
  edm::ESHandle<EcalTPGSlidingWindow> theEcalTPGSlidingWindow_handle;
  setup.get<EcalTPGSlidingWindowRcd>().get(theEcalTPGSlidingWindow_handle);
  const EcalTPGSlidingWindow *ecaltpgSlidW = theEcalTPGSlidingWindow_handle.product();
  edm::ESHandle<EcalTPGWeightIdMap> theEcalTPGWEightIdMap_handle;
  setup.get<EcalTPGWeightIdMapRcd>().get(theEcalTPGWEightIdMap_handle);
  const EcalTPGWeightIdMap *ecaltpgWeightMap = theEcalTPGWEightIdMap_handle.product();
  edm::ESHandle<EcalTPGWeightGroup> theEcalTPGWEightGroup_handle;
  setup.get<EcalTPGWeightGroupRcd>().get(theEcalTPGWEightGroup_handle);
  const EcalTPGWeightGroup *ecaltpgWeightGroup = theEcalTPGWEightGroup_handle.product();
  edm::ESHandle<EcalTPGFineGrainStripEE> theEcalTPGFineGrainStripEE_handle;
  setup.get<EcalTPGFineGrainStripEERcd>().get(theEcalTPGFineGrainStripEE_handle);
  const EcalTPGFineGrainStripEE *ecaltpgFgStripEE = theEcalTPGFineGrainStripEE_handle.product();
  edm::ESHandle<EcalTPGStripStatus> theEcalTPGStripStatus_handle;
  setup.get<EcalTPGStripStatusRcd>().get(theEcalTPGStripStatus_handle);
  const EcalTPGStripStatus *ecaltpgStripStatus = theEcalTPGStripStatus_handle.product();

  algo_->setPointers(ecaltpLin,
                     ecaltpPed,
                     ecaltpgSlidW,
                     ecaltpgWeightMap,
                     ecaltpgWeightGroup,
                     ecaltpgFgStripEE,
                     ecaltpgBadX,
                     ecaltpgStripStatus);

  // .. and for EcalFenixTcp
  // get parameter records for towers
  edm::ESHandle<EcalTPGFineGrainEBGroup> theEcalTPGFineGrainEBGroup_handle;
  setup.get<EcalTPGFineGrainEBGroupRcd>().get(theEcalTPGFineGrainEBGroup_handle);
  const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup = theEcalTPGFineGrainEBGroup_handle.product();

  edm::ESHandle<EcalTPGLutGroup> theEcalTPGLutGroup_handle;
  setup.get<EcalTPGLutGroupRcd>().get(theEcalTPGLutGroup_handle);
  const EcalTPGLutGroup *ecaltpgLutGroup = theEcalTPGLutGroup_handle.product();

  edm::ESHandle<EcalTPGLutIdMap> theEcalTPGLutIdMap_handle;
  setup.get<EcalTPGLutIdMapRcd>().get(theEcalTPGLutIdMap_handle);
  const EcalTPGLutIdMap *ecaltpgLut = theEcalTPGLutIdMap_handle.product();

  edm::ESHandle<EcalTPGFineGrainEBIdMap> theEcalTPGFineGrainEBIdMap_handle;
  setup.get<EcalTPGFineGrainEBIdMapRcd>().get(theEcalTPGFineGrainEBIdMap_handle);
  const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB = theEcalTPGFineGrainEBIdMap_handle.product();

  edm::ESHandle<EcalTPGFineGrainTowerEE> theEcalTPGFineGrainTowerEE_handle;
  setup.get<EcalTPGFineGrainTowerEERcd>().get(theEcalTPGFineGrainTowerEE_handle);
  const EcalTPGFineGrainTowerEE *ecaltpgFineGrainTowerEE = theEcalTPGFineGrainTowerEE_handle.product();

  edm::ESHandle<EcalTPGTowerStatus> theEcalTPGTowerStatus_handle;
  setup.get<EcalTPGTowerStatusRcd>().get(theEcalTPGTowerStatus_handle);
  const EcalTPGTowerStatus *ecaltpgBadTT = theEcalTPGTowerStatus_handle.product();

  edm::ESHandle<EcalTPGSpike> theEcalTPGSpike_handle;
  setup.get<EcalTPGSpikeRcd>().get(theEcalTPGSpike_handle);
  const EcalTPGSpike *ecaltpgSpike = theEcalTPGSpike_handle.product();

  algo_->setPointers2(ecaltpgFgEBGroup,
                      ecaltpgLutGroup,
                      ecaltpgLut,
                      ecaltpgFineGrainEB,
                      ecaltpgFineGrainTowerEE,
                      ecaltpgBadTT,
                      ecaltpgSpike);

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
    algo_->run(iSetup, ebdc, *pOut, *pOutTcp);
  }

  if (endcap) {
    eedc = eeDigis.product();
    algo_->run(iSetup, eedc, *pOut, *pOutTcp);
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
  descriptions.addDefault(desc);
}
