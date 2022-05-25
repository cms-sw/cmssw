// framework
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// for reconstruction
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

// geometry
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

// my include files
#include "SimG4Core/GFlash/TB/TreeMatrixCalib.h"

// root includes
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TSelector.h"
#include "TApplication.h"

// c++ includes
#include <string>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <math.h>
#include <memory>
#include <stdexcept>

class TreeProducerCalibSimul : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TreeProducerCalibSimul(const edm::ParameterSet&);
  ~TreeProducerCalibSimul() override = default;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override;
  void endJob() override;

private:
  std::string rootfile_;
  std::string txtfile_;
  std::string EBRecHitCollection_;
  std::string RecHitProducer_;
  std::string hodoRecInfoCollection_;
  std::string hodoRecInfoProducer_;
  std::string tdcRecInfoCollection_;
  std::string tdcRecInfoProducer_;
  std::string eventHeaderCollection_;
  std::string eventHeaderProducer_;
  double posCluster_;

  std::unique_ptr<TreeMatrixCalib> myTree_;

  edm::EDGetTokenT<EBRecHitCollection> tokEBRecHit_;
  edm::EDGetTokenT<EcalTBHodoscopeRecInfo> tokEcalHodo_;
  edm::EDGetTokenT<EcalTBTDCRecInfo> tokEcalTDC_;
  edm::EDGetTokenT<EcalTBEventHeader> tokEventHeader_;

  int xtalInBeam_;
  int tot_events_;
  int tot_events_ok_;
  int noHits_;
  int noHodo_;
  int noTdc_;
  int noHeader_;
};

// -------------------------------------------------
// contructor
TreeProducerCalibSimul::TreeProducerCalibSimul(const edm::ParameterSet& iConfig) {
  usesResource(TFileService::kSharedResource);

  // now do what ever initialization is needed
  xtalInBeam_ = iConfig.getUntrackedParameter<int>("xtalInBeam", -1000);
  rootfile_ = iConfig.getUntrackedParameter<std::string>("rootfile", "mySimMatrixTree.root");
  txtfile_ = iConfig.getUntrackedParameter<std::string>("txtfile", "mySimMatrixTree.txt");
  EBRecHitCollection_ = iConfig.getParameter<std::string>("EBRecHitCollection");
  RecHitProducer_ = iConfig.getParameter<std::string>("RecHitProducer");
  hodoRecInfoCollection_ = iConfig.getParameter<std::string>("hodoRecInfoCollection");
  hodoRecInfoProducer_ = iConfig.getParameter<std::string>("hodoRecInfoProducer");
  tdcRecInfoCollection_ = iConfig.getParameter<std::string>("tdcRecInfoCollection");
  tdcRecInfoProducer_ = iConfig.getParameter<std::string>("tdcRecInfoProducer");
  eventHeaderCollection_ = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_ = iConfig.getParameter<std::string>("eventHeaderProducer");

  // summary
  edm::LogVerbatim("GFlash") << "\nConstructor\n\nTreeProducerCalibSimul\nxtal in beam = " << xtalInBeam_;
  edm::LogVerbatim("GFlash") << "Fetching hitCollection: " << EBRecHitCollection_.c_str() << " prod by "
                             << RecHitProducer_.c_str();
  edm::LogVerbatim("GFlash") << "Fetching hodoCollection: " << hodoRecInfoCollection_.c_str() << " prod by "
                             << hodoRecInfoProducer_.c_str();
  edm::LogVerbatim("GFlash") << "Fetching tdcCollection: " << tdcRecInfoCollection_.c_str() << " prod by "
                             << tdcRecInfoProducer_.c_str();
  edm::LogVerbatim("GFlash") << "Fetching evHeaCollection: " << eventHeaderCollection_.c_str() << " prod by "
                             << eventHeaderProducer_.c_str() << "\n";

  tokEBRecHit_ = consumes<EBRecHitCollection>(edm::InputTag(RecHitProducer_, EBRecHitCollection_));
  tokEcalHodo_ = consumes<EcalTBHodoscopeRecInfo>(edm::InputTag(hodoRecInfoProducer_, hodoRecInfoCollection_));
  tokEcalTDC_ = consumes<EcalTBTDCRecInfo>(edm::InputTag(tdcRecInfoProducer_, tdcRecInfoCollection_));
  tokEventHeader_ = consumes<EcalTBEventHeader>(edm::InputTag(eventHeaderProducer_));
}

// ------------------------------------------------------
// initializations
void TreeProducerCalibSimul::beginJob() {
  edm::LogVerbatim("GFlash") << "\nBeginJob\n";

  // tree
  myTree_ = std::make_unique<TreeMatrixCalib>(rootfile_.c_str());

  // counters
  tot_events_ = 0;
  tot_events_ok_ = 0;
  noHits_ = 0;
  noHodo_ = 0;
  noTdc_ = 0;
  noHeader_ = 0;
}

// -------------------------------------------
// finalizing
void TreeProducerCalibSimul::endJob() {
  edm::LogVerbatim("GFlash") << "\nEndJob\n";

  std::ofstream MyOut(txtfile_.c_str());
  MyOut << "total events: " << tot_events_ << std::endl;
  MyOut << "events skipped because of no hits: " << noHits_ << std::endl;
  MyOut << "events skipped because of no hodos: " << noHodo_ << std::endl;
  MyOut << "events skipped because of no tdc: " << noTdc_ << std::endl;
  MyOut << "events skipped because of no header: " << noHeader_ << std::endl;
  MyOut << "total OK events (passing the basic selection): " << tot_events_ok_ << std::endl;
  MyOut.close();
}

// -----------------------------------------------
// my analysis
void TreeProducerCalibSimul::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // counting events
  ++tot_events_;

  if (tot_events_ % 5000 == 0)
    edm::LogVerbatim("GFlash") << "event " << tot_events_;

  // ---------------------------------------------------------------------
  // taking what I need: hits
  const EBRecHitCollection* EBRecHits = &iEvent.get(tokEBRecHit_);

  // taking what I need: hodoscopes
  const EcalTBHodoscopeRecInfo* recHodo = &iEvent.get(tokEcalHodo_);

  // taking what I need: tdc
  const EcalTBTDCRecInfo* recTDC = &iEvent.get(tokEcalTDC_);

  // taking what I need: event header
  const EcalTBEventHeader* evtHeader = &iEvent.get(tokEventHeader_);

  // checking everything is there and fine
  if ((!EBRecHits) || (EBRecHits->size() == 0)) {
    ++noHits_;
    return;
  }
  if (!recTDC) {
    ++noTdc_;
    return;
  }
  if (!recHodo) {
    ++noHodo_;
    return;
  }
  if (!evtHeader) {
    ++noHeader_;
    return;
  }
  ++tot_events_ok_;

  // ---------------------------------------------------------------------
  // info on the event
  int run = -999;
  int tbm = -999;
  int event = evtHeader->eventNumber();

  // ---------------------------------------------------------------------
  // xtal-in-beam
  int nomXtalInBeam = -999;
  int nextXtalInBeam = -999;

  EBDetId xtalInBeamId(1, xtalInBeam_, EBDetId::SMCRYSTALMODE);
  if (xtalInBeamId == EBDetId(0)) {
    return;
  }
  int mySupCry = xtalInBeamId.ic();
  int mySupEta = xtalInBeamId.ieta();
  int mySupPhi = xtalInBeamId.iphi();

  // ---------------------------------------------------------------------
  // hodoscope information
  double x = recHodo->posX();
  double y = recHodo->posY();
  double sx = recHodo->slopeX();
  double sy = recHodo->slopeY();
  double qx = recHodo->qualX();
  double qy = recHodo->qualY();

  // ---------------------------------------------------------------------
  // tdc information
  double tdcOffset = recTDC->offset();

  // ---------------------------------------------------------------------
  // Find EBDetId in a 7x7 Matrix
  EBDetId Xtals7x7[49];
  double energy[49];
  int crystal[49];
  int allMatrix = 1;
  for (unsigned int icry = 0; icry < 49; icry++) {
    unsigned int row = icry / 7;
    unsigned int column = icry % 7;
    Xtals7x7[icry] = EBDetId(xtalInBeamId.ieta() + column - 3, xtalInBeamId.iphi() + row - 3, EBDetId::ETAPHIMODE);

    if (Xtals7x7[icry].ism() == 1) {
      energy[icry] = EBRecHits->find(Xtals7x7[icry])->energy();
      crystal[icry] = Xtals7x7[icry].ic();
    } else {
      energy[icry] = -100.;
      crystal[icry] = -100;
      allMatrix = 0;
    }
  }

  // ---------------------------------------------------------------------
  // Looking for the max energy crystal
  double maxEne = -999.;
  int maxEneCry = 9999;
  for (int ii = 0; ii < 49; ii++) {
    if (energy[ii] > maxEne) {
      maxEne = energy[ii];
      maxEneCry = crystal[ii];
    }
  }

  // Position reconstruction - skipped here
  double Xcal = -999.;
  double Ycal = -999.;

  // filling the tree
  myTree_->fillInfo(run,
                    event,
                    mySupCry,
                    maxEneCry,
                    nomXtalInBeam,
                    nextXtalInBeam,
                    mySupEta,
                    mySupPhi,
                    tbm,
                    x,
                    y,
                    Xcal,
                    Ycal,
                    sx,
                    sy,
                    qx,
                    qy,
                    tdcOffset,
                    allMatrix,
                    energy,
                    crystal);
  myTree_->store();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in

DEFINE_FWK_MODULE(TreeProducerCalibSimul);
