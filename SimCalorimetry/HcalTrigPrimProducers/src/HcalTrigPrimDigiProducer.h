#ifndef HcalTrigPrimDigiProducer_h
#define HcalTrigPrimDigiProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalTriggerPrimitiveAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalLutMetadata.h"
#include "CondFormats/DataRecord/interface/HcalLutMetadataRcd.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

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

#endif
