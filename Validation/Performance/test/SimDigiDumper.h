#ifndef SimDigiDumper_H
#define SimDigiDumper_H
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
// ecal calorimeter info
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
// hcal calorimeter info
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
// silicon strip info
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
// silicon pixel info
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
// muon CSC Strip info
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
// muon CSC Wire info
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
// muon RPC info
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include <vector>

class SimDigiDumper : public edm::EDAnalyzer{
public:
  explicit SimDigiDumper( const edm::ParameterSet& );
  virtual ~SimDigiDumper() {};

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void beginJob(){};
  virtual void endJob(){};
private:

  edm::EDGetTokenT<EBDigiCollection> ECalEBSrc_;
  edm::EDGetTokenT<EEDigiCollection> ECalEESrc_;
  edm::EDGetTokenT<ESDigiCollection> ECalESSrc_;

  edm::EDGetTokenT<HBHEDigiCollection> HCalDigi_;
  edm::EDGetTokenT<HODigiCollection> HCalHODigi_;
  edm::EDGetTokenT<HFDigiCollection> HCalHFDigi_;

  edm::EDGetTokenT<ZDCDigiCollection> ZdcDigi_;

  edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > SiStripSrc_;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > SiPxlSrc_;

  edm::EDGetTokenT<DTDigiCollection> MuDTSrc_;

  edm::EDGetTokenT<CSCStripDigiCollection> MuCSCStripSrc_;

  edm::EDGetTokenT<CSCWireDigiCollection> MuCSCWireSrc_;

  edm::EDGetTokenT<RPCDigiCollection> MuRPCSrc_;

  static const int sdSiTIB          = 3;
  static const int sdSiTID          = 4;
  static const int sdSiTOB          = 5;
  static const int sdSiTEC          = 6;
  static const int sdPxlBrl         = 1;
  static const int sdPxlFwd         = 2;


};

#endif
