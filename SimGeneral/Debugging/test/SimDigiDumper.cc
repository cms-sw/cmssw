// system include files
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
// ecal calorimeter info
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
// hcal calorimeter info
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
// silicon strip info
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
// silicon pixel info
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
// muon DT info
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
// muon CSC Strip info
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
// muon CSC Wire info
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
// muon RPC info
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
// BTL/ETL info
#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"

#include "FWCore/Framework/interface/MakerMacros.h"

class SimDigiDumper : public edm::one::EDAnalyzer<> {
public:
  explicit SimDigiDumper(const edm::ParameterSet&);
  virtual ~SimDigiDumper(){};

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(){};
  virtual void endJob(){};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<EBDigiCollection> ECalEBSrc_;
  edm::EDGetTokenT<EEDigiCollection> ECalEESrc_;
  edm::EDGetTokenT<ESDigiCollection> ECalESSrc_;

  edm::EDGetTokenT<HBHEDigiCollection> HCalSrc_;
  edm::EDGetTokenT<HODigiCollection> HCalHOSrc_;
  edm::EDGetTokenT<HFDigiCollection> HCalHFSrc_;

  edm::EDGetTokenT<ZDCDigiCollection> ZdcSrc_;

  edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > SiStripSrc_;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > SiPxlSrc_;

  edm::EDGetTokenT<DTDigiCollection> MuDTSrc_;

  edm::EDGetTokenT<CSCStripDigiCollection> MuCSCStripSrc_;

  edm::EDGetTokenT<CSCWireDigiCollection> MuCSCWireSrc_;

  edm::EDGetTokenT<RPCDigiCollection> MuRPCSrc_;

  edm::EDGetTokenT<BTLDigiCollection> BTLSrc_;
  edm::EDGetTokenT<ETLDigiCollection> ETLSrc_;

  static const int sdSiTIB = 3;
  static const int sdSiTID = 4;
  static const int sdSiTOB = 5;
  static const int sdSiTEC = 6;
  static const int sdPxlBrl = 1;
  static const int sdPxlFwd = 2;
};

SimDigiDumper::SimDigiDumper(const edm::ParameterSet& iPSet) {
  //get Labels to use to extract information
  ECalEBSrc_ = consumes<EBDigiCollection>(iPSet.getParameter<edm::InputTag>("ECalEBSrc"));
  ECalEESrc_ = consumes<EEDigiCollection>(iPSet.getParameter<edm::InputTag>("ECalEESrc"));
  ECalESSrc_ = consumes<ESDigiCollection>(iPSet.getParameter<edm::InputTag>("ECalESSrc"));
  HCalSrc_ = consumes<HBHEDigiCollection>(iPSet.getParameter<edm::InputTag>("HCalSrc"));
  HCalHOSrc_ = consumes<HODigiCollection>(iPSet.getParameter<edm::InputTag>("HCalSrc"));
  HCalHFSrc_ = consumes<HFDigiCollection>(iPSet.getParameter<edm::InputTag>("HCalSrc"));
  ZdcSrc_ = consumes<ZDCDigiCollection>(iPSet.getParameter<edm::InputTag>("ZdcSrc"));
  SiStripSrc_ = consumes<edm::DetSetVector<SiStripDigi> >(iPSet.getParameter<edm::InputTag>("SiStripSrc"));
  SiPxlSrc_ = consumes<edm::DetSetVector<PixelDigi> >(iPSet.getParameter<edm::InputTag>("SiPxlSrc"));
  MuDTSrc_ = consumes<DTDigiCollection>(iPSet.getParameter<edm::InputTag>("MuDTSrc"));
  MuCSCStripSrc_ = consumes<CSCStripDigiCollection>(iPSet.getParameter<edm::InputTag>("MuCSCStripSrc"));
  MuCSCWireSrc_ = consumes<CSCWireDigiCollection>(iPSet.getParameter<edm::InputTag>("MuCSCWireSrc"));
  MuRPCSrc_ = consumes<RPCDigiCollection>(iPSet.getParameter<edm::InputTag>("MuRPCSrc"));
  BTLSrc_ = consumes<BTLDigiCollection>(iPSet.getParameter<edm::InputTag>("BTLSrc"));
  ETLSrc_ = consumes<ETLDigiCollection>(iPSet.getParameter<edm::InputTag>("ETLSrc"));
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void SimDigiDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // ECAL Barrel

  bool isBarrel = true;
  const EBDigiCollection* EBdigis = 0;
  auto EcalDigiEB = iEvent.getHandle(ECalEBSrc_);
  if (!EcalDigiEB.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find EcalDigiEB in event!";
  } else {
    EBdigis = EcalDigiEB.product();
    if (EcalDigiEB->size() == 0)
      isBarrel = false;
    edm::LogPrint("SimDigiDumper") << "Ecal Barrel, digi multiplicity = " << EcalDigiEB->size();

    if (isBarrel) {
      // loop over digis
      for (unsigned int digis = 0; digis < EcalDigiEB->size(); ++digis) {
        EBDataFrame ebdf = (*EBdigis)[digis];
        edm::LogPrint("SimDigiDumper") << ebdf;
      }
    }
  }

  // ECAL Endcap
  bool isEndcap = true;
  const EEDigiCollection* EEdigis = 0;
  auto EcalDigiEE = iEvent.getHandle(ECalEESrc_);
  if (!EcalDigiEE.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find EcalDigiEE in event!";
  } else {
    EEdigis = EcalDigiEE.product();
    if (EcalDigiEE->size() == 0)
      isEndcap = false;
    edm::LogPrint("SimDigiDumper") << "Ecal Endcap, digi multiplicity = " << EcalDigiEE->size();

    if (isEndcap) {
      // loop over digis
      for (unsigned int digis = 0; digis < EcalDigiEE->size(); ++digis) {
        EEDataFrame eedf = (*EEdigis)[digis];
        edm::LogPrint("SimDigiDumper") << eedf;
      }
    }
  }

  // ECAL Preshower
  bool isPreshower = true;
  const ESDigiCollection* ESdigis = 0;
  auto EcalDigiES = iEvent.getHandle(ECalESSrc_);
  if (!EcalDigiES.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find EcalDigiES in event!";
  } else {
    ESdigis = EcalDigiES.product();
    if (EcalDigiES->size() == 0)
      isPreshower = false;
    edm::LogPrint("SimDigiDumper") << "Ecal Preshower, digi multiplicity = " << EcalDigiES->size();

    if (isPreshower) {
      // loop over digis
      for (unsigned int digis = 0; digis < EcalDigiES->size(); ++digis) {
        ESDataFrame esdf = (*ESdigis)[digis];
        edm::LogPrint("SimDigiDumper") << esdf;
      }
    }
  }

  // HBHE
  bool isHBHE = true;
  const HBHEDigiCollection* HBHEdigis = 0;
  auto hbhe = iEvent.getHandle(HCalSrc_);
  if (!hbhe.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find HBHEDataFrame in event!";
  } else {
    HBHEdigis = hbhe.product();
    if (hbhe->size() == 0)
      isHBHE = false;
    edm::LogPrint("SimDigiDumper") << "HBHE, digi multiplicity = " << hbhe->size();

    if (isHBHE) {
      //loop over digis
      for (unsigned int digis = 0; digis < hbhe->size(); ++digis) {
        HBHEDataFrame hehbdf = (*HBHEdigis)[digis];
        edm::LogPrint("SimDigiDumper") << hehbdf;
        //edm::SortedCollection<HBHEDataFrame>::const_iterator ihbhe;
        //for  (ihbhe == hbhe->begin(); ihbhe != hbhe->end(); ihbhe++) {
        //edm::LogPrint("SimDigiDumper") << "Nothing" ;
        //edm::LogPrint("SimDigiDumper") << (*ihbhe) ;
      }
    }
  }

  // HO
  bool isHO = true;
  const HODigiCollection* HOdigis = 0;
  auto ho = iEvent.getHandle(HCalHOSrc_);
  if (!ho.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find HODataFrame in event!";
  } else {
    HOdigis = ho.product();
    if (ho->size() == 0)
      isHO = false;
    edm::LogPrint("SimDigiDumper") << "HO, digi multiplicity = " << ho->size();

    if (isHO) {
      //loop over digis
      for (unsigned int digis = 0; digis < ho->size(); ++digis) {
        HODataFrame hodf = (*HOdigis)[digis];
        edm::LogPrint("SimDigiDumper") << hodf;
      }
    }
  }

  // HF
  bool isHF = true;
  const HFDigiCollection* HFdigis = 0;
  auto hf = iEvent.getHandle(HCalHFSrc_);
  if (!hf.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find HFDataFrame in event!";
  } else {
    HFdigis = hf.product();
    if (hf->size() == 0)
      isHF = false;
    edm::LogPrint("SimDigiDumper") << "HF, digi multiplicity = " << hf->size();

    if (isHF) {
      //loop over digis
      for (unsigned int digis = 0; digis < hf->size(); ++digis) {
        HFDataFrame hodf = (*HFdigis)[digis];
        edm::LogPrint("SimDigiDumper") << hodf;
      }
    }
  }

  // ZDC
  bool isZDC = true;
  const ZDCDigiCollection* ZDCdigis = 0;
  auto zdc = iEvent.getHandle(ZdcSrc_);
  if (!zdc.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find ZDCDataFrame in event!";
  } else {
    ZDCdigis = zdc.product();
    if (zdc->size() == 0)
      isZDC = false;
    edm::LogPrint("SimDigiDumper") << "ZDC, digi multiplicity = " << zdc->size();

    if (isZDC) {
      //loop over digis
      for (unsigned int digis = 0; digis < zdc->size(); ++digis) {
        ZDCDataFrame hodf = (*ZDCdigis)[digis];
        edm::LogPrint("SimDigiDumper") << hodf;
      }
    }
  }

  // Strip Tracker
  bool isStrip = true;
  auto stripDigis = iEvent.getHandle(SiStripSrc_);
  if (!stripDigis.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find stripDigis in event!";
  } else {
    if (stripDigis->size() == 0)
      isStrip = false;
    edm::LogPrint("SimDigiDumper") << "Strip Tracker, digi multiplicity = " << stripDigis->size();
    if (isStrip) {
      edm::DetSetVector<SiStripDigi>::const_iterator DSViter;
      for (DSViter = stripDigis->begin(); DSViter != stripDigis->end(); ++DSViter) {
        edm::DetSet<SiStripDigi>::const_iterator begin = DSViter->data.begin();
        edm::DetSet<SiStripDigi>::const_iterator end = DSViter->data.end();
        edm::DetSet<SiStripDigi>::const_iterator iter;
        unsigned int id = DSViter->id;
        DetId detId(id);

        if (detId.subdetId() == sdSiTIB) {
          edm::LogPrint("SimDigiDumper") << "TIB " << DSViter->data.size();
        } else if (detId.subdetId() == sdSiTOB) {
          edm::LogPrint("SimDigiDumper") << "TOB " << DSViter->data.size();
        } else if (detId.subdetId() == sdSiTID) {
          edm::LogPrint("SimDigiDumper") << "TID " << DSViter->data.size();
        }
        if (detId.subdetId() == sdSiTEC) {
          edm::LogPrint("SimDigiDumper") << "TEC " << DSViter->data.size();
        }
        for (iter = begin; iter != end; ++iter) {
          edm::LogPrint("SimDigiDumper") << (*iter);
        }
      }
    }
  }

  // Pixel Tracker
  bool isPixel = true;
  auto pixelDigis = iEvent.getHandle(SiPxlSrc_);
  if (!pixelDigis.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find pixelDigis in event!";
  } else {
    if (pixelDigis->size() == 0)
      isPixel = false;
    edm::LogPrint("SimDigiDumper") << "Pixel Tracker, digi multiplicity = " << pixelDigis->size();

    if (isPixel) {
      edm::DetSetVector<PixelDigi>::const_iterator DSViter;
      for (DSViter = pixelDigis->begin(); DSViter != pixelDigis->end(); ++DSViter) {
        edm::DetSet<PixelDigi>::const_iterator begin = DSViter->data.begin();
        edm::DetSet<PixelDigi>::const_iterator end = DSViter->data.end();
        edm::DetSet<PixelDigi>::const_iterator iter;
        unsigned int id = DSViter->id;
        DetId detId(id);

        if (detId.subdetId() == sdPxlBrl) {
          edm::LogPrint("SimDigiDumper") << "Pixel barrel " << DSViter->data.size();
        } else if (detId.subdetId() == sdPxlFwd) {
          edm::LogPrint("SimDigiDumper") << "Pixel forward " << DSViter->data.size();
        }
        for (iter = begin; iter != end; ++iter) {
          edm::LogPrint("SimDigiDumper") << (*iter);
        }
      }
    }
  }

  // DT
  bool isDT = true;
  auto dtDigis = iEvent.getHandle(MuDTSrc_);
  if (!dtDigis.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find dtDigis in event!";
  }
  unsigned int nDT = 0;
  if (dtDigis->begin() == dtDigis->end()) {
    isDT = false;
    edm::LogPrint("SimDigiDumper") << "dtDigis seem empty!";
  }
  if (isDT) {
    DTDigiCollection::DigiRangeIterator dtLayerIt;
    for (dtLayerIt = dtDigis->begin(); dtLayerIt != dtDigis->end(); ++dtLayerIt) {
      const DTDigiCollection::Range& range = (*dtLayerIt).second;
      edm::LogPrint("SimDigiDumper") << "DT layer = " << (*dtLayerIt).first << " digi ";
      for (DTDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
        edm::LogPrint("SimDigiDumper") << (*digiIt);
        nDT++;
      }
    }
  }
  edm::LogPrint("SimDigiDumper") << "DT, digi multiplicity = " << nDT;

  // CSC strip
  bool isCSCStrip = true;
  auto cscStripDigis = iEvent.getHandle(MuCSCStripSrc_);
  if (!cscStripDigis.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find cscStripDigis in event!";
  }
  if (cscStripDigis->begin() == cscStripDigis->end())
    isCSCStrip = false;
  unsigned int nCSCStrip = 0;

  if (isCSCStrip) {
    CSCStripDigiCollection::DigiRangeIterator detUnitIt;
    for (detUnitIt = cscStripDigis->begin(); detUnitIt != cscStripDigis->end(); ++detUnitIt) {
      const CSCStripDigiCollection::Range& range = (*detUnitIt).second;
      edm::LogPrint("SimDigiDumper") << "CSC detid = " << (*detUnitIt).first << " digi ";
      for (CSCStripDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
        edm::LogPrint("SimDigiDumper") << (*digiIt);
        nCSCStrip++;
      }
    }
  }
  edm::LogPrint("SimDigiDumper") << "CSC strip, digi multiplicity = " << nCSCStrip;

  // CSC wire
  bool isCSCWire = true;
  auto cscWireDigis = iEvent.getHandle(MuCSCWireSrc_);
  if (!cscWireDigis.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find cscWireDigis in event!";
  }
  if (cscWireDigis->begin() == cscWireDigis->end())
    isCSCWire = false;
  unsigned int nCSCWire = 0;

  if (isCSCWire) {
    CSCWireDigiCollection::DigiRangeIterator detUnitIt;
    for (detUnitIt = cscWireDigis->begin(); detUnitIt != cscWireDigis->end(); ++detUnitIt) {
      const CSCWireDigiCollection::Range& range = (*detUnitIt).second;
      edm::LogPrint("SimDigiDumper") << "CSC detid = " << (*detUnitIt).first << " digi ";
      for (CSCWireDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
        edm::LogPrint("SimDigiDumper") << (*digiIt);
        nCSCWire++;
      }
    }
  }
  edm::LogPrint("SimDigiDumper") << "CSC wire, digi multiplicity = " << nCSCWire;

  // RPC
  bool isRPC = true;
  auto rpcDigis = iEvent.getHandle(MuRPCSrc_);
  if (!rpcDigis.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find rpcDigis in event!";
  }
  if (rpcDigis->begin() == rpcDigis->end())
    isRPC = false;
  unsigned int nRPC = 0;

  if (isRPC) {
    RPCDigiCollection::DigiRangeIterator detUnitIt;
    for (detUnitIt = rpcDigis->begin(); detUnitIt != rpcDigis->end(); ++detUnitIt) {
      const RPCDigiCollection::Range& range = (*detUnitIt).second;
      edm::LogPrint("SimDigiDumper") << "RPC detid = " << (*detUnitIt).first << " digi ";
      for (RPCDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
        edm::LogPrint("SimDigiDumper") << (*digiIt);
        nRPC++;
      }
    }
  }
  edm::LogPrint("SimDigiDumper") << "RPC, digi multiplicity = " << nRPC;

  // BTL
  bool isBTL = true;
  const BTLDigiCollection* BTLdigis = 0;
  auto BTLDigi = iEvent.getHandle(BTLSrc_);
  if (!BTLDigi.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find BTLDigi in event!";
  } else {
    BTLdigis = BTLDigi.product();
    if (BTLDigi->size() == 0)
      isBTL = false;
    edm::LogPrint("SimDigiDumper") << "Barrel Timing Layer, digi multiplicity = " << BTLDigi->size();

    if (isBTL) {
      // loop over digis
      for (unsigned int digis = 0; digis < BTLDigi->size(); ++digis) {
        BTLDataFrame btldf = (*BTLdigis)[digis];
        edm::LogPrint("SimDigiDumper") << btldf.id().rawId();
        btldf.print();
      }
    }
  }

  // ETL
  bool isETL = true;
  const ETLDigiCollection* ETLdigis = 0;
  auto ETLDigi = iEvent.getHandle(ETLSrc_);
  if (!ETLDigi.isValid()) {
    edm::LogPrint("SimDigiDumper") << "Unable to find ETLDigi in event!";
  } else {
    ETLdigis = ETLDigi.product();
    if (ETLDigi->size() == 0)
      isETL = false;
    edm::LogPrint("SimDigiDumper") << "Endcap Timing Layer, digi multiplicity = " << ETLDigi->size();

    if (isETL) {
      // loop over digis
      for (unsigned int digis = 0; digis < ETLDigi->size(); ++digis) {
        ETLDataFrame etldf = (*ETLdigis)[digis];
        edm::LogPrint("SimDigiDumper") << etldf.id().rawId();
        etldf.print();
      }
    }
  }

  return;
}

void SimDigiDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ECalEBSrc", edm::InputTag("simEcalDigis", "ebDigis"))->setComment("ECAL barrel digis");
  desc.add<edm::InputTag>("ECalEESrc", edm::InputTag("simEcalDigis", "eeDigis"))->setComment("ECAL endcap digis");
  desc.add<edm::InputTag>("ECalESSrc", edm::InputTag("simEcalPreshowerDigis"))->setComment("ECAL preshower digis");
  desc.add<edm::InputTag>("HCalSrc", edm::InputTag("simHcalDigis"))->setComment("HCAL digis");
  desc.add<edm::InputTag>("ZdcSrc", edm::InputTag("simHcalUnsuppressedDigis"))->setComment("ZDC digis");
  desc.add<edm::InputTag>("SiStripSrc", edm::InputTag("simSiStripDigis", "ZeroSuppressed"))
      ->setComment("Si strip digis");
  desc.add<edm::InputTag>("SiPxlSrc", edm::InputTag("simSiPixelDigis"))->setComment("Si pixel digis");
  desc.add<edm::InputTag>("MuDTSrc", edm::InputTag("simMuonDTDigis"))->setComment("DT digis");
  desc.add<edm::InputTag>("MuCSCStripSrc", edm::InputTag("simMuonCSCDigis", "MuonCSCStripDigi"))
      ->setComment("CSC strip digis");
  desc.add<edm::InputTag>("MuCSCWireSrc", edm::InputTag("simMuonCSCDigis", "MuonCSCWireDigi"))
      ->setComment("CSC wire digis");
  desc.add<edm::InputTag>("MuRPCSrc", edm::InputTag("simMuonRPCDigis"))->setComment("RPC digis");
  desc.add<edm::InputTag>("BTLSrc", edm::InputTag("mix", "FTLBarrel"))->setComment("BTL digis");
  desc.add<edm::InputTag>("ETLSrc", edm::InputTag("mix", "FTLEndcap"))->setComment("ETL digis");
  descriptions.add("simDigiDumper", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimDigiDumper);
