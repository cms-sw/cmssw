
#include "SimCalorimetry/EcalZeroSuppressionProducers/interface/EcalZeroSuppressionProducer.h"

EcalZeroSuppressionProducer::EcalZeroSuppressionProducer(const edm::ParameterSet &params)
    : glbBarrelThreshold_(params.getUntrackedParameter<double>("glbBarrelThreshold", 0.2)),
      glbEndcapThreshold_(params.getUntrackedParameter<double>("glbEndcapThreshold", 0.4)),
      digiProducer_(params.getParameter<std::string>("digiProducer")),
      ebDigiCollection_(params.getParameter<std::string>("EBdigiCollection")),
      eeDigiCollection_(params.getParameter<std::string>("EEdigiCollection")),
      ebZSdigiCollection_(params.getParameter<std::string>("EBZSdigiCollection")),
      eeZSdigiCollection_(params.getParameter<std::string>("EEZSdigiCollection")),
      ebToken_(consumes<EBDigiCollection>(edm::InputTag(digiProducer_))),
      eeToken_(consumes<EEDigiCollection>(edm::InputTag(digiProducer_))),
      pedestalToken_(esConsumes()) {
  produces<EBDigiCollection>(ebZSdigiCollection_);
  produces<EEDigiCollection>(eeZSdigiCollection_);
}

EcalZeroSuppressionProducer::~EcalZeroSuppressionProducer() {}

void EcalZeroSuppressionProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  // Get Inputs

  initCalibrations(eventSetup);

  edm::Handle<EBDigiCollection> pEBDigis;
  edm::Handle<EEDigiCollection> pEEDigis;

  const EBDigiCollection *fullBarrelDigis = nullptr;
  const EEDigiCollection *fullEndcapDigis = nullptr;

  event.getByToken(ebToken_, pEBDigis);
  if (pEBDigis.isValid()) {
    fullBarrelDigis = pEBDigis.product();  // get a ptr to the produc
    edm::LogInfo("ZeroSuppressionInfo") << "total # fullBarrelDigis: " << fullBarrelDigis->size();
  } else {
    edm::LogError("ZeroSuppressionError") << "Error! can't get the product " << ebDigiCollection_.c_str();
  }

  event.getByToken(eeToken_, pEEDigis);
  if (pEEDigis.isValid()) {
    fullEndcapDigis = pEEDigis.product();  // get a ptr to the product
    edm::LogInfo("ZeroSuppressionInfo") << "total # fullEndcapDigis: " << fullEndcapDigis->size();
  } else {
    edm::LogError("ZeroSuppressionError") << "Error! can't get the product " << eeDigiCollection_.c_str();
  }

  // collection of zero suppressed digis to put in the event

  auto gzsBarrelDigis = std::make_unique<EBDigiCollection>();
  auto gzsEndcapDigis = std::make_unique<EEDigiCollection>();

  CaloDigiCollectionSorter sorter(5);

  // Barrel zero suppression

  if (fullBarrelDigis) {
    for (EBDigiCollection::const_iterator digiItr = (*fullBarrelDigis).begin(); digiItr != (*fullBarrelDigis).end();
         ++digiItr) {
      bool isAccepted = theBarrelZeroSuppressor_.accept(*digiItr, glbBarrelThreshold_);
      if (isAccepted) {
        (*gzsBarrelDigis).push_back(digiItr->id(), digiItr->begin());
      }
    }
    edm::LogInfo("ZeroSuppressionInfo") << "EB Digis: " << gzsBarrelDigis->size();

    // std::vector<EBDataFrame> sortedDigisEB =
    // sorter.sortedVector(*gzsBarrelDigis); LogDebug("ZeroSuppressionDump") <<
    // "Top 10 EB digis"; for(int i = 0; i < std::min(10,(int)
    // sortedDigisEB.size()); ++i)
    //  {
    //    LogDebug("ZeroSuppressionDump") << sortedDigisEB[i];
    //  }
  }

  // Endcap zero suppression

  if (fullEndcapDigis) {
    for (EEDigiCollection::const_iterator digiItr = (*fullEndcapDigis).begin(); digiItr != (*fullEndcapDigis).end();
         ++digiItr) {
      bool isAccepted = theEndcapZeroSuppressor_.accept(*digiItr, glbEndcapThreshold_);
      if (isAccepted) {
        (*gzsEndcapDigis).push_back(digiItr->id(), digiItr->begin());
      }
    }
    edm::LogInfo("ZeroSuppressionInfo") << "EB Digis: " << gzsBarrelDigis->size();

    //    std::vector<EEDataFrame> sortedDigisEE =
    //    sorter.sortedVector(*gzsEndcapDigis);
    // LogDebug("ZeroSuppressionDump")  << "Top 10 EE digis";
    // for(int i = 0; i < std::min(10,(int) sortedDigisEE.size()); ++i)
    //  {
    //    LogDebug("ZeroSuppressionDump") << sortedDigisEE[i];
    //  }
  }
  // Step D: Put outputs into event
  event.put(std::move(gzsBarrelDigis), ebZSdigiCollection_);
  event.put(std::move(gzsEndcapDigis), eeZSdigiCollection_);
}

void EcalZeroSuppressionProducer::initCalibrations(const edm::EventSetup &eventSetup) {
  // Pedestals from event setup
  const auto &thePedestals = eventSetup.getData(pedestalToken_);

  theBarrelZeroSuppressor_.setPedestals(&thePedestals);
  theEndcapZeroSuppressor_.setPedestals(&thePedestals);
}
