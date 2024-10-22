#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/DataRecord/interface/EcalLiteDTUPedestalsRcd.h"

#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGAmplWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGTimeWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

// We keep these lines for future posssible necessary additions
//#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
//#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"
//#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TrigPrimAlgo.h"
#include <memory>

// Class declaration
/** \class EcalEBTrigPrimPhase2Producer                                                                                                                                                   \author L. Lutton, N. Marinelli - Univ. of Notre Dame                                                                                                                                   Description: forPhase II                                                                                                                                                               It consumes the new Phase2 digis based on the new EB electronics                                                                                                                       and plugs in the main steering algo for TP emulation                                                                                                                                   It produces the EcalEBPhase2TrigPrimDigiCollection                                                                                                                                 */

class EcalEBPhase2TrigPrimAlgo;

class EcalEBTrigPrimPhase2Producer : public edm::stream::EDProducer<> {
public:
  explicit EcalEBTrigPrimPhase2Producer(const edm::ParameterSet& conf);

  ~EcalEBTrigPrimPhase2Producer() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::unique_ptr<EcalEBPhase2TrigPrimAlgo> algo_;
  bool debug_;
  bool famos_;
  int nEvent_;
  edm::EDGetTokenT<EBDigiCollectionPh2> tokenEBdigi_;
  edm::ESGetToken<EcalEBPhase2TPGLinearizationConst, EcalEBPhase2TPGLinearizationConstRcd>
      theEcalEBPhase2TPGLinearization_Token_;
  edm::ESGetToken<EcalEBPhase2TPGPedestalsMap, EcalEBPhase2TPGPedestalsRcd> theEcalEBPhase2TPGPedestals_Token_;

  edm::ESGetToken<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd> theEcalTPGPedestals_Token_;

  edm::ESGetToken<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd> theEcalTPGCrystalStatus_Token_;
  edm::ESGetToken<EcalEBPhase2TPGAmplWeightIdMap, EcalEBPhase2TPGAmplWeightIdMapRcd> theEcalEBTPGAmplWeightIdMap_Token_;
  edm::ESGetToken<EcalEBPhase2TPGTimeWeightIdMap, EcalEBPhase2TPGTimeWeightIdMapRcd> theEcalEBTPGTimeWeightIdMap_Token_;

  edm::ESGetToken<EcalTPGWeightGroup, EcalTPGWeightGroupRcd> theEcalTPGWeightGroup_Token_;

  edm::ESGetToken<EcalTPGTowerStatus, EcalTPGTowerStatusRcd> theEcalTPGTowerStatus_Token_;
  edm::ESGetToken<EcalTPGSpike, EcalTPGSpikeRcd> theEcalTPGSpike_Token_;

  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> eTTmapToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> theGeometryToken_;

  int binOfMaximum_;
  bool fillBinOfMaximumFromHistory_;

  unsigned long long getRecords(edm::EventSetup const& setup);
  unsigned long long cacheID_;
};

EcalEBTrigPrimPhase2Producer::EcalEBTrigPrimPhase2Producer(const edm::ParameterSet& iConfig)
    : debug_(iConfig.getParameter<bool>("Debug")),
      famos_(iConfig.getParameter<bool>("Famos")),
      binOfMaximum_(iConfig.getParameter<int>("binOfMaximum")) {
  tokenEBdigi_ = consumes<EBDigiCollectionPh2>(iConfig.getParameter<edm::InputTag>("barrelEcalDigis"));

  eTTmapToken_ = esConsumes<edm::Transition::BeginRun>();
  theGeometryToken_ = esConsumes<edm::Transition::BeginRun>();

  theEcalTPGPedestals_Token_ =
      esConsumes<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd, edm::Transition::BeginRun>();
  theEcalEBPhase2TPGPedestals_Token_ =
      esConsumes<EcalEBPhase2TPGPedestalsMap, EcalEBPhase2TPGPedestalsRcd, edm::Transition::BeginRun>();

  theEcalTPGCrystalStatus_Token_ =
      esConsumes<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd, edm::Transition::BeginRun>();
  theEcalEBPhase2TPGLinearization_Token_ =
      esConsumes<EcalEBPhase2TPGLinearizationConst, EcalEBPhase2TPGLinearizationConstRcd, edm::Transition::BeginRun>();
  theEcalEBTPGAmplWeightIdMap_Token_ =
      esConsumes<EcalEBPhase2TPGAmplWeightIdMap, EcalEBPhase2TPGAmplWeightIdMapRcd, edm::Transition::BeginRun>();
  theEcalEBTPGTimeWeightIdMap_Token_ =
      esConsumes<EcalEBPhase2TPGTimeWeightIdMap, EcalEBPhase2TPGTimeWeightIdMapRcd, edm::Transition::BeginRun>();
  theEcalTPGWeightGroup_Token_ = esConsumes<EcalTPGWeightGroup, EcalTPGWeightGroupRcd, edm::Transition::BeginRun>();

  //register your products
  produces<EcalEBPhase2TrigPrimDigiCollection>();
}

void EcalEBTrigPrimPhase2Producer::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  auto const& theGeometry = setup.getData(theGeometryToken_);
  auto const& eTTmap = setup.getData(eTTmapToken_);

  algo_ = std::make_unique<EcalEBPhase2TrigPrimAlgo>(&eTTmap, &theGeometry, binOfMaximum_, debug_);

  // get a first version of the records
  cacheID_ = this->getRecords(setup);

  nEvent_ = 0;
}

void EcalEBTrigPrimPhase2Producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("Debug", false);
  desc.add<bool>("Famos", false);
  desc.add<int>("BinOfMaximum", 6);  // this needs to be at the same value used for the Phase2 LiteDTU digis !
  desc.add<edm::InputTag>("barrelEcalDigis", edm::InputTag("simEcalUnsuppressedDigis"));
}

unsigned long long EcalEBTrigPrimPhase2Producer::getRecords(edm::EventSetup const& setup) {
  // get parameter records for xtals
  auto theEcalEBPhase2TPGLinearization_handle = setup.getHandle(theEcalEBPhase2TPGLinearization_Token_);
  const EcalEBPhase2TPGLinearizationConst* ecaltpLin = theEcalEBPhase2TPGLinearization_handle.product();
  //
  edm::ESHandle<EcalLiteDTUPedestalsMap> theEcalTPGPedestals_handle = setup.getHandle(theEcalTPGPedestals_Token_);
  const EcalLiteDTUPedestalsMap* ecaltpPed = theEcalTPGPedestals_handle.product();
  //
  // auto theEcalEBPhase2TPGPedestals_handle = setup.getHandle(theEcalEBPhase2TPGPedestals_Token_);
  //const EcalEBPhase2TPGPedestalsMap* ebTPPedestals = theEcalEBPhase2TPGPedestals_handle.product();
  //
  edm::ESHandle<EcalTPGCrystalStatus> theEcalTPGCrystalStatus_handle = setup.getHandle(theEcalTPGCrystalStatus_Token_);
  const EcalTPGCrystalStatus* ecaltpgBadX = theEcalTPGCrystalStatus_handle.product();
  //
  edm::ESHandle<EcalEBPhase2TPGAmplWeightIdMap> theEcalEBTPGAmplWeightIdMap_handle =
      setup.getHandle(theEcalEBTPGAmplWeightIdMap_Token_);
  const EcalEBPhase2TPGAmplWeightIdMap* ecaltpgAmplWeightMap = theEcalEBTPGAmplWeightIdMap_handle.product();
  //
  edm::ESHandle<EcalEBPhase2TPGTimeWeightIdMap> theEcalEBTPGTimeWeightIdMap_handle =
      setup.getHandle(theEcalEBTPGTimeWeightIdMap_Token_);
  const EcalEBPhase2TPGTimeWeightIdMap* ecaltpgTimeWeightMap = theEcalEBTPGTimeWeightIdMap_handle.product();
  //
  edm::ESHandle<EcalTPGWeightGroup> theEcalTPGWeightGroup_handle = setup.getHandle(theEcalTPGWeightGroup_Token_);
  const EcalTPGWeightGroup* ecaltpgWeightGroup = theEcalTPGWeightGroup_handle.product();
  // These commented out lines are for reminder for possible needed implementations
  //edm::ESHandle<EcalTPGTowerStatus> theEcalTPGTowerStatus_handle = setup.getHandle(theEcalTPGTowerStatus_Token_);
  //const EcalTPGTowerStatus* ecaltpgBadTT = theEcalTPGTowerStatus_handle.product();
  //
  //edm::ESHandle<EcalTPGSpike> theEcalTPGSpike_handle = setup.getHandle(theEcalTPGSpike_Token_);
  //const EcalTPGSpike* ecaltpgSpike = theEcalTPGSpike_handle.product();

  ////////////////
  algo_->setPointers(ecaltpPed, ecaltpLin, ecaltpgBadX, ecaltpgAmplWeightMap, ecaltpgTimeWeightMap, ecaltpgWeightGroup);

  return setup.get<EcalEBPhase2TPGLinearizationConstRcd>().cacheIdentifier();
  //  return setup.get<EcalLiteDTUPedestalsRcd>().cacheIdentifier();
}

void EcalEBTrigPrimPhase2Producer::endRun(edm::Run const& run, edm::EventSetup const& setup) { algo_.reset(); }

EcalEBTrigPrimPhase2Producer::~EcalEBTrigPrimPhase2Producer() {}

// ------------ method called to produce the data  ------------
void EcalEBTrigPrimPhase2Producer::produce(edm::Event& e, const edm::EventSetup& iSetup) {
  nEvent_++;

  // get input collections
  edm::Handle<EBDigiCollectionPh2> barrelDigiHandle;

  if (!e.getByToken(tokenEBdigi_, barrelDigiHandle)) {
    edm::EDConsumerBase::Labels labels;
    labelsForToken(tokenEBdigi_, labels);
    edm::LogWarning("EcalTPG") << " Couldnt find Barrel digis " << labels.module << " and label "
                               << labels.productInstance << "!!!";
  }
  const auto* ebdigi = barrelDigiHandle.product();

  if (debug_)
    LogDebug("EcalEBTrigPrimPhase2Producer")
        << " EcalTPG"
        << " =================> Treating event  " << nEvent_ << ", Number of EB digis "
        << barrelDigiHandle.product()->size() << std::endl;

  auto pOut = std::make_unique<EcalEBPhase2TrigPrimDigiCollection>();

  // invoke algorithm
  algo_->run(ebdigi, *pOut);

  if (debug_) {
    LogDebug("EcalEBTrigPrimPhase2Producer")
        << "produce"
        << " For Barrel  " << pOut->size() << " TP  Digis were produced" << std::endl;
  }

  //  debug prints if TP > 0. The number of TP with Et>0 is also used later for a LogInfo
  int nonZeroTP = 0;
  int nXstal = 0;
  for (unsigned int i = 0; i < pOut->size(); ++i) {
    nXstal++;
    for (int isam = 0; isam < (*pOut)[i].size(); ++isam) {
      if ((*pOut)[i][isam].encodedEt() > 0) {
        nonZeroTP++;
        if (debug_) {
          LogDebug("EcalEBTrigPrimPhase2Producer")
              << " For xStal n " << nXstal << " xTsal Id " << (((*pOut)[i])).id() << ", TP is " << (*pOut)[i]
              << " (*pOut)[i][isam].raw() " << (*pOut)[i][isam].raw() << "  (*pOut)[i][isam].encodedEt() "
              << (*pOut)[i][isam].encodedEt() << "  (*pOut)[i][isam].time() " << (*pOut)[i][isam].time() << std::endl;
        }
      }
    }
  }

  edm::LogInfo("EcalEBTrigPrimPhase2Producer")
      << "EcalTPG"
      << "\n =================> For Barrel , " << pOut->size() << " TP  Digis were produced (including zero ones)"
      << " Non zero primitives were " << nonZeroTP << std::endl;

  // put result into the Event
  e.put(std::move(pOut));
}

DEFINE_FWK_MODULE(EcalEBTrigPrimPhase2Producer);
