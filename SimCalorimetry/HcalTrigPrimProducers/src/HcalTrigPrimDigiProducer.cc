#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTrigPrimDigiProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <algorithm>




HcalTrigPrimDigiProducer::HcalTrigPrimDigiProducer(const edm::ParameterSet& ps)
: 
  theAlgo_(ps.getParameter<bool>("peakFilter"),
        ps.getParameter<std::vector<double> >("weights"),
        ps.getParameter<int>("latency"),
        ps.getParameter<uint32_t>("FG_threshold"),
	ps.getParameter<std::vector<uint32_t> >("FG_HF_thresholds"),
        ps.getParameter<uint32_t>("ZS_threshold"),
        ps.getParameter<int>("numberOfSamples"),
        ps.getParameter<int>("numberOfPresamples"),
        ps.getParameter<int>("numberOfSamplesHF"),
        ps.getParameter<int>("numberOfPresamplesHF"),
        ps.getParameter<bool>("useTDCInMinBiasBits"),
        ps.getParameter<uint32_t>("MinSignalThreshold"),
        ps.getParameter<uint32_t>("PMTNoiseThreshold")
  ),
  inputLabel_(ps.getParameter<std::vector<edm::InputTag> >("inputLabel")),
  inputUpgradeLabel_(ps.getParameter<std::vector<edm::InputTag> >("inputUpgradeLabel")),
  inputTagFEDRaw_(ps.getParameter<edm::InputTag> ("InputTagFEDRaw")),
  runZS_(ps.getParameter<bool>("RunZS")),
  runFrontEndFormatError_(ps.getParameter<bool>("FrontEndFormatError"))
{
   std::vector<bool> upgrades = {ps.getParameter<bool>("upgradeHB"), ps.getParameter<bool>("upgradeHE"), ps.getParameter<bool>("upgradeHF")};
   upgrade_ = std::any_of(std::begin(upgrades), std::end(upgrades), [](bool a) { return a; });
   legacy_ = std::any_of(std::begin(upgrades), std::end(upgrades), [](bool a) { return !a; });

   if (ps.exists("parameters")) {
      auto pset = ps.getUntrackedParameter<edm::ParameterSet>("parameters");
      theAlgo_.overrideParameters(pset);
   }
   theAlgo_.setUpgradeFlags(upgrades[0], upgrades[1], upgrades[2]);

    HFEMB_ = false;
    if(ps.exists("LSConfig"))
    {
        LongShortCut_ = ps.getUntrackedParameter<edm::ParameterSet>("LSConfig");
        HFEMB_ = LongShortCut_.getParameter<bool>("HcalFeatureHFEMBit");
        MinLongEnergy_ = LongShortCut_.getParameter<double>("Min_Long_Energy"); //minimum long energy
        MinShortEnergy_ = LongShortCut_.getParameter<double>("Min_Short_Energy"); //minimum short energy
        LongShortSlope_ = LongShortCut_.getParameter<double>("Long_vrs_Short_Slope"); //slope of the line that cuts are based on
        LongShortOffset_ = LongShortCut_.getParameter<double>("Long_Short_Offset"); //offset of line
    }
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

   produces<HcalTrigPrimDigiCollection>();
   theAlgo_.setPeakFinderAlgorithm(ps.getParameter<int>("PeakFinderAlgorithm"));

   edm::ParameterSet hfSS=ps.getParameter<edm::ParameterSet>("tpScales").getParameter<edm::ParameterSet>("HF");

   theAlgo_.setNCTScaleShift(hfSS.getParameter<int>("NCTShift"));
   theAlgo_.setRCTScaleShift(hfSS.getParameter<int>("RCTShift"));
}


void HcalTrigPrimDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& eventSetup) {

  // Step A: get the conditions, for the decoding
  edm::ESHandle<HcalTPGCoder> inputCoder;
  eventSetup.get<HcalTPGRecord>().get(inputCoder);

  edm::ESHandle<CaloTPGTranscoder> outTranscoder;
  eventSetup.get<CaloTPGRecord>().get(outTranscoder);

  edm::ESHandle<HcalLutMetadata> lutMetadata;
  eventSetup.get<HcalLutMetadataRcd>().get(lutMetadata);
  float rctlsb = lutMetadata->getRctLsb();

  edm::ESHandle<HcalTrigTowerGeometry> pG;
  eventSetup.get<CaloGeometryRecord>().get(pG);
  
  // Step B: Create empty output
  std::unique_ptr<HcalTrigPrimDigiCollection> result(new HcalTrigPrimDigiCollection());

  edm::Handle<HBHEDigiCollection> hbheDigis;
  edm::Handle<HFDigiCollection>   hfDigis;

  edm::Handle<QIE11DigiCollection> hbheUpDigis;
  edm::Handle<QIE10DigiCollection> hfUpDigis;

  if (legacy_) {
     iEvent.getByToken(tok_hbhe_,hbheDigis);
     iEvent.getByToken(tok_hf_,hfDigis);

     // protect here against missing input collections
     // there is no protection in HcalTriggerPrimitiveAlgo

     if (!hbheDigis.isValid() and legacy_) {
         edm::LogInfo("HcalTrigPrimDigiProducer")
                 << "\nWarning: HBHEDigiCollection with input tag "
                 << inputLabel_[0]
                 << "\nrequested in configuration, but not found in the event."
                 << "\nQuit returning empty product." << std::endl;

         // put empty HcalTrigPrimDigiCollection in the event
         iEvent.put(std::move(result));

         return;
     }

     if (!hfDigis.isValid() and legacy_) {
         edm::LogInfo("HcalTrigPrimDigiProducer")
                 << "\nWarning: HFDigiCollection with input tag "
                 << inputLabel_[1]
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
                 << "\nWarning: Upgrade HBHEDigiCollection with input tag "
                 << inputUpgradeLabel_[0]
                 << "\nrequested in configuration, but not found in the event."
                 << "\nQuit returning empty product." << std::endl;

         // put empty HcalTrigPrimDigiCollection in the event
         iEvent.put(std::move(result));

         return;
     }

     if (!hfUpDigis.isValid() and upgrade_) {
         edm::LogInfo("HcalTrigPrimDigiProducer")
                 << "\nWarning: HFDigiCollection with input tag "
                 << inputUpgradeLabel_[1]
                 << "\nrequested in configuration, but not found in the event."
                 << "\nQuit returning empty product." << std::endl;

         // put empty HcalTrigPrimDigiCollection in the event
         iEvent.put(std::move(result));

         return;
     }
  }


    edm::ESHandle < HcalDbService > pSetup;
    eventSetup.get<HcalDbRecord> ().get(pSetup);

    HcalFeatureBit* hfembit = nullptr;

    if(HFEMB_)
    {
        hfembit = new HcalFeatureHFEMBit(MinShortEnergy_, MinLongEnergy_, LongShortSlope_, LongShortOffset_, *pSetup); //inputs values that cut will be based on
    }

  // Step C: Invoke the algorithm, passing in inputs and getting back outputs.
  if (legacy_ and not upgrade_) {
     theAlgo_.run(inputCoder.product(), outTranscoder->getHcalCompressor().get(), pSetup.product(),
           *result, &(*pG), rctlsb, hfembit, *hbheDigis, *hfDigis);
  } else if (legacy_ and upgrade_) {
     theAlgo_.run(inputCoder.product(), outTranscoder->getHcalCompressor().get(), pSetup.product(),
           *result, &(*pG), rctlsb, hfembit, *hbheDigis, *hfDigis, *hbheUpDigis, *hfUpDigis);
  } else {
     theAlgo_.run(inputCoder.product(), outTranscoder->getHcalCompressor().get(), pSetup.product(),
           *result, &(*pG), rctlsb, hfembit, *hbheUpDigis, *hfUpDigis);
  }


  // Step C.1: Run FE Format Error / ZS for real data.
  if (runFrontEndFormatError_) {

       
        const HcalElectronicsMap *emap = pSetup->getHcalMapping();

        edm::Handle < FEDRawDataCollection > fedHandle;
        iEvent.getByToken(tok_raw_, fedHandle);

        if (fedHandle.isValid() && emap != nullptr) {
            theAlgo_.runFEFormatError(fedHandle.product(), emap, *result);
        } else {
            edm::LogInfo("HcalTrigPrimDigiProducer")
                    << "\nWarning: FEDRawDataCollection with input tag "
                    << inputTagFEDRaw_
                    << "\nrequested in configuration, but not found in the event."
                    << "\nQuit returning empty product." << std::endl;

            // produce empty HcalTrigPrimDigiCollection and put it in the event
            std::unique_ptr < HcalTrigPrimDigiCollection > emptyResult(
                    new HcalTrigPrimDigiCollection());

            iEvent.put(std::move(emptyResult));

            return;
        }

  }

  if (runZS_) theAlgo_.runZS(*result);

  //  edm::LogInfo("HcalTrigPrimDigiProducer") << "HcalTrigPrims: " << result->size();

  // Step D: Put outputs into event
  iEvent.put(std::move(result));
}


