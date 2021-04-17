/** \class EcalEBTrigPrimProducer
 * For Phase II
 * EcalEBTrigPrimProducer produces a EcalEBTrigPrimDigiCollection
 * out of PhaseI Digis. This is a simple starting point to fill in the chain
 * for Phase II
 * 
 *
 *
 ************************************************************/
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//

/*
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainEBIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
*/

#include "EcalEBTrigPrimProducer.h"

#include <memory>

#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimTestAlgo.h"

EcalEBTrigPrimProducer::EcalEBTrigPrimProducer(const edm::ParameterSet& iConfig)
    : barrelOnly_(iConfig.getParameter<bool>("BarrelOnly")),
      tcpFormat_(iConfig.getParameter<bool>("TcpOutput")),
      debug_(iConfig.getParameter<bool>("Debug")),
      famos_(iConfig.getParameter<bool>("Famos")),
      nSamples_(iConfig.getParameter<int>("nOfSamples")),
      binOfMaximum_(iConfig.getParameter<int>("binOfMaximum")) {
  tokenEBdigi_ = consumes<EBDigiCollection>(iConfig.getParameter<edm::InputTag>("barrelEcalDigis"));
  theEcalTPGLinearization_Token_ =
      esConsumes<EcalTPGLinearizationConst, EcalTPGLinearizationConstRcd, edm::Transition::BeginRun>();
  theEcalTPGPedestals_Token_ = esConsumes<EcalTPGPedestals, EcalTPGPedestalsRcd, edm::Transition::BeginRun>();
  theEcalTPGCrystalStatus_Token_ =
      esConsumes<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd, edm::Transition::BeginRun>();
  theEcalTPGWEightIdMap_Token_ = esConsumes<EcalTPGWeightIdMap, EcalTPGWeightIdMapRcd, edm::Transition::BeginRun>();
  theEcalTPGWEightGroup_Token_ = esConsumes<EcalTPGWeightGroup, EcalTPGWeightGroupRcd, edm::Transition::BeginRun>();
  theEcalTPGSlidingWindow_Token_ =
      esConsumes<EcalTPGSlidingWindow, EcalTPGSlidingWindowRcd, edm::Transition::BeginRun>();
  theEcalTPGLutGroup_Token_ = esConsumes<EcalTPGLutGroup, EcalTPGLutGroupRcd, edm::Transition::BeginRun>();
  theEcalTPGLutIdMap_Token_ = esConsumes<EcalTPGLutIdMap, EcalTPGLutIdMapRcd, edm::Transition::BeginRun>();
  theEcalTPGTowerStatus_Token_ = esConsumes<EcalTPGTowerStatus, EcalTPGTowerStatusRcd, edm::Transition::BeginRun>();
  theEcalTPGSpike_Token_ = esConsumes<EcalTPGSpike, EcalTPGSpikeRcd, edm::Transition::BeginRun>();
  //register your products
  produces<EcalEBTrigPrimDigiCollection>();
  if (tcpFormat_)
    produces<EcalEBTrigPrimDigiCollection>("formatTCP");
  if (not barrelOnly_) {
    eTTmapToken_ = esConsumes<edm::Transition::BeginRun>();
    theGeometryToken_ = esConsumes<edm::Transition::BeginRun>();
  }
}

void EcalEBTrigPrimProducer::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  //ProcessHistory is guaranteed to be constant for an entire Run
  //binOfMaximum_ = findBinOfMaximum(fillBinOfMaximumFromHistory_,binOfMaximum_,run.processHistory());

  if (barrelOnly_) {
    algo_ = std::make_unique<EcalEBTrigPrimTestAlgo>(nSamples_, binOfMaximum_, tcpFormat_, debug_, famos_);
  } else {
    auto const& theGeometry = setup.getData(theGeometryToken_);
    auto const& eTTmap = setup.getData(eTTmapToken_);
    algo_ = std::make_unique<EcalEBTrigPrimTestAlgo>(
        &eTTmap, &theGeometry, nSamples_, binOfMaximum_, tcpFormat_, debug_, famos_);
  }
  // get a first version of the records
  cacheID_ = this->getRecords(setup);
  nEvent_ = 0;
}

unsigned long long EcalEBTrigPrimProducer::getRecords(edm::EventSetup const& setup) {
  // get parameter records for xtals
  auto theEcalTPGLinearization_handle = setup.getHandle(theEcalTPGLinearization_Token_);
  const EcalTPGLinearizationConst* ecaltpLin = theEcalTPGLinearization_handle.product();
  //
  edm::ESHandle<EcalTPGPedestals> theEcalTPGPedestals_handle = setup.getHandle(theEcalTPGPedestals_Token_);
  const EcalTPGPedestals* ecaltpPed = theEcalTPGPedestals_handle.product();
  //
  edm::ESHandle<EcalTPGCrystalStatus> theEcalTPGCrystalStatus_handle = setup.getHandle(theEcalTPGCrystalStatus_Token_);
  const EcalTPGCrystalStatus* ecaltpgBadX = theEcalTPGCrystalStatus_handle.product();
  //
  //for strips
  //
  edm::ESHandle<EcalTPGWeightIdMap> theEcalTPGWEightIdMap_handle = setup.getHandle(theEcalTPGWEightIdMap_Token_);
  const EcalTPGWeightIdMap* ecaltpgWeightMap = theEcalTPGWEightIdMap_handle.product();
  //
  edm::ESHandle<EcalTPGWeightGroup> theEcalTPGWEightGroup_handle = setup.getHandle(theEcalTPGWEightGroup_Token_);
  const EcalTPGWeightGroup* ecaltpgWeightGroup = theEcalTPGWEightGroup_handle.product();
  //
  edm::ESHandle<EcalTPGSlidingWindow> theEcalTPGSlidingWindow_handle = setup.getHandle(theEcalTPGSlidingWindow_Token_);
  const EcalTPGSlidingWindow* ecaltpgSlidW = theEcalTPGSlidingWindow_handle.product();
  //  TCP
  edm::ESHandle<EcalTPGLutGroup> theEcalTPGLutGroup_handle = setup.getHandle(theEcalTPGLutGroup_Token_);
  const EcalTPGLutGroup* ecaltpgLutGroup = theEcalTPGLutGroup_handle.product();
  //
  edm::ESHandle<EcalTPGLutIdMap> theEcalTPGLutIdMap_handle = setup.getHandle(theEcalTPGLutIdMap_Token_);
  const EcalTPGLutIdMap* ecaltpgLut = theEcalTPGLutIdMap_handle.product();
  //
  edm::ESHandle<EcalTPGTowerStatus> theEcalTPGTowerStatus_handle = setup.getHandle(theEcalTPGTowerStatus_Token_);
  const EcalTPGTowerStatus* ecaltpgBadTT = theEcalTPGTowerStatus_handle.product();
  //
  edm::ESHandle<EcalTPGSpike> theEcalTPGSpike_handle = setup.getHandle(theEcalTPGSpike_Token_);
  const EcalTPGSpike* ecaltpgSpike = theEcalTPGSpike_handle.product();

  ////////////////
  algo_->setPointers(ecaltpLin,
                     ecaltpPed,
                     ecaltpgBadX,
                     ecaltpgWeightMap,
                     ecaltpgWeightGroup,
                     ecaltpgSlidW,
                     ecaltpgLutGroup,
                     ecaltpgLut,
                     ecaltpgBadTT,
                     ecaltpgSpike);
  return setup.get<EcalTPGLinearizationConstRcd>().cacheIdentifier();
}

void EcalEBTrigPrimProducer::endRun(edm::Run const& run, edm::EventSetup const& setup) { algo_.reset(); }

EcalEBTrigPrimProducer::~EcalEBTrigPrimProducer() {}

// ------------ method called to produce the data  ------------
void EcalEBTrigPrimProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) {
  nEvent_++;

  // get input collections
  edm::Handle<EBDigiCollection> barrelDigiHandle;

  if (!e.getByToken(tokenEBdigi_, barrelDigiHandle)) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(tokenEBdigi_, labels);
    edm::LogWarning("EcalTPG") << " Couldnt find Barrel digis " << labels.module << " and label "
                               << labels.productInstance << "!!!";
  }

  if (debug_)
    std::cout << "EcalTPG"
              << " =================> Treating event  " << nEvent_ << ", Number of EB digis "
              << barrelDigiHandle.product()->size() << std::endl;

  auto pOut = std::make_unique<EcalEBTrigPrimDigiCollection>();
  auto pOutTcp = std::make_unique<EcalEBTrigPrimDigiCollection>();

  // if ( e.id().event() != 648 ) return;

  //std::cout << " Event number " << e.id().event() << std::endl;

  // invoke algorithm

  const EBDigiCollection* ebdigi = nullptr;
  ebdigi = barrelDigiHandle.product();
  algo_->run(ebdigi, *pOut, *pOutTcp);

  if (debug_)
    std::cout << "produce"
              << " For Barrel  " << pOut->size() << " TP  Digis were produced" << std::endl;

  //  debug prints if TP >0

  int nonZeroTP = 0;
  for (unsigned int i = 0; i < pOut->size(); ++i) {
    if (debug_) {
      std::cout << "EcalTPG Printing only non zero TP "
                << " For tower  " << (((*pOut)[i])).id() << ", TP is " << (*pOut)[i];
      for (int isam = 0; isam < (*pOut)[i].size(); ++isam) {
        if ((*pOut)[i][isam].encodedEt() > 0) {
          nonZeroTP++;
          std::cout << " (*pOut)[i][isam].raw() " << (*pOut)[i][isam].raw() << "  (*pOut)[i][isam].encodedEt() "
                    << (*pOut)[i][isam].encodedEt() << std::endl;
        }
      }
    }
  }
  if (debug_)
    std::cout << "EcalTPG"
              << "\n =================> For Barrel , " << pOut->size()
              << " TP  Digis were produced (including zero ones)"
              << " Non zero primitives were " << nonZeroTP << std::endl;

  // put result into the Event
  e.put(std::move(pOut));
  if (tcpFormat_)
    e.put(std::move(pOutTcp), "formatTCP");
}
