#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"

/*TP Code*/
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
/*~TP Code*/

#include <map>
#include <memory>
#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "TH2D.h"
#include "TH1D.h"
#include "TProfile.h"

//#define EDM_ML_DEBUG

class HcalDigiStudy : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HcalDigiStudy(const edm::ParameterSet&);
  ~HcalDigiStudy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginJob() override {}
  void endJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  struct HistLim {
    HistLim(int nbin, double mini, double maxi) : n(nbin), min(mini), max(maxi) {}
    int n;
    double min;
    double max;
  };

  std::map<std::string, TH1D*> hist1_;
  std::map<std::string, TH2D*> hist2_;
  std::map<std::string, TProfile*> histP_;

  void book1D(edm::Service<TFileService>& fs, std::string name, int n, double min, double max);
  void book1D(edm::Service<TFileService>& fs, std::string name, const HistLim& limX);
  void book2D(edm::Service<TFileService>& fs, std::string name, const HistLim& limX, const HistLim& limY);
  void bookPf(edm::Service<TFileService>& fs, std::string name, const HistLim& limX, const HistLim& limY);
  void bookPf(
      edm::Service<TFileService>& fs, std::string name, const HistLim& limX, const HistLim& limY, const char* option);
  void booking(edm::Service<TFileService>& fs, std::string subdetopt, int bnoise, int bmc);

  void fill1D(std::string name, double X, double weight = 1);
  void fill2D(std::string name, double X, double Y, double weight = 1);
  void fillPf(std::string name, double X, double Y);

  std::string str(int x);

  template <class Digi>
  void reco(const edm::Event& iEvent,
            const edm::EventSetup& iSetup,
            const edm::EDGetTokenT<edm::SortedCollection<Digi> >& tok);
  template <class dataFrameType>
  void reco(const edm::Event& iEvent,
            const edm::EventSetup& iSetup,
            const edm::EDGetTokenT<HcalDataFrameContainer<dataFrameType> >& tok);

  std::string outputFile_;
  std::string subdet_;
  std::string zside_;
  //    std::string inputLabel_;
  edm::InputTag inputTag_;
  edm::InputTag QIE10inputTag_;
  edm::InputTag QIE11inputTag_;
  edm::InputTag emulTPsTag_;
  edm::InputTag dataTPsTag_;
  std::string mode_;
  std::string mc_;
  int noise_;
  bool testNumber_;
  bool hep17_;
  bool HEPhase1_;
  bool HBPhase1_;
  bool Plot_TP_ver_;

  edm::EDGetTokenT<edm::PCaloHitContainer> tok_mc_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_emulTPs_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_dataTPs_;

  edm::EDGetTokenT<QIE10DigiCollection> tok_qie10_hf_;
  edm::EDGetTokenT<QIE11DigiCollection> tok_qie11_hbhe_;

  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_Geom_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> tok_Decoder_;
  edm::ESGetToken<HcalTrigTowerGeometry, CaloGeometryRecord> tok_TPGeom_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_Topo_;
  edm::ESGetToken<HcalDbService, HcalDbRecord> tok_Cond_;

  const HcalDbService* conditions_;
  const HcalDDDRecConstants* hcons_;
  const HcalTopology* htopo_;

  int nevent1;
  int nevent2;
  int nevent3;
  int nevent4;
  int nevtot;

  int maxDepth_[5];   // 0:any, 1:HB, 2:HE, 3:HF
  int nChannels_[5];  // 0:any, 1:HB, 2:HE,

  bool skipDataTPs;
  bool skipTPs;
};

HcalDigiStudy::HcalDigiStudy(const edm::ParameterSet& iConfig) {
  usesResource(TFileService::kSharedResource);

  subdet_ = iConfig.getUntrackedParameter<std::string>("subdetector", "all");
  inputTag_ = iConfig.getParameter<edm::InputTag>("digiTag");
  QIE10inputTag_ = iConfig.getParameter<edm::InputTag>("QIE10digiTag");
  QIE11inputTag_ = iConfig.getParameter<edm::InputTag>("QIE11digiTag");
  emulTPsTag_ = iConfig.getParameter<edm::InputTag>("emulTPs");
  dataTPsTag_ = iConfig.getParameter<edm::InputTag>("dataTPs");
  mc_ = iConfig.getUntrackedParameter<std::string>("mc", "no");
  mode_ = iConfig.getUntrackedParameter<std::string>("mode", "multi");
  testNumber_ = iConfig.getParameter<bool>("TestNumber");
  hep17_ = iConfig.getParameter<bool>("hep17");
  HEPhase1_ = iConfig.getParameter<bool>("HEPhase1");
  HBPhase1_ = iConfig.getParameter<bool>("HBPhase1");
  Plot_TP_ver_ = iConfig.getParameter<bool>("Plot_TP_ver");

  // register for data access
  if (iConfig.exists("simHits")) {
    tok_mc_ = consumes<edm::PCaloHitContainer>(iConfig.getUntrackedParameter<edm::InputTag>("simHits"));
  }
  tok_hbhe_ = consumes<HBHEDigiCollection>(inputTag_);
  tok_ho_ = consumes<HODigiCollection>(inputTag_);
  tok_hf_ = consumes<HFDigiCollection>(inputTag_);
  tok_emulTPs_ = consumes<HcalTrigPrimDigiCollection>(emulTPsTag_);
  if (dataTPsTag_ == edm::InputTag("")) {
    skipDataTPs = true;
    skipTPs = true;
  } else {
    skipDataTPs = false;
    tok_dataTPs_ = consumes<HcalTrigPrimDigiCollection>(dataTPsTag_);
    skipTPs = false;
  }

  tok_qie10_hf_ = consumes<QIE10DigiCollection>(QIE10inputTag_);
  tok_qie11_hbhe_ = consumes<QIE11DigiCollection>(QIE11inputTag_);

  tok_HRNDC_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  tok_Geom_ = esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>();
  tok_Decoder_ = esConsumes<CaloTPGTranscoder, CaloTPGRecord>();
  tok_TPGeom_ = esConsumes<HcalTrigTowerGeometry, CaloGeometryRecord>();
  tok_Topo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_Cond_ = esConsumes<HcalDbService, HcalDbRecord>();

  nevent1 = 0;
  nevent2 = 0;
  nevent3 = 0;
  nevent4 = 0;
  nevtot = 0;
}

void HcalDigiStudy::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("QIE10digiTag", edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("QIE11digiTag", edm::InputTag("hcalDigis"));
  desc.addUntracked<std::string>("mode", "multi");
  desc.addUntracked<std::string>("hcalselector", "all");
  desc.addUntracked<std::string>("mc", "yes");
  desc.addUntracked<edm::InputTag>("simHits", edm::InputTag("g4SimHits", "HcalHits"));
  desc.add<edm::InputTag>("emulTPs", edm::InputTag("emulDigis"));
  desc.add<edm::InputTag>("dataTPs", edm::InputTag(""));
  desc.add<bool>("TestNumber", false);
  desc.add<bool>("hep17", false);
  desc.add<bool>("HEPhase1", false);
  desc.add<bool>("HBPhase1", false);
  desc.add<bool>("Plot_TP_ver", false);
  descriptions.add("hcalDigiStudy", desc);
}

void HcalDigiStudy::beginRun(const edm::Run& run, const edm::EventSetup& es) {
  hcons_ = &es.getData(tok_HRNDC_);

  maxDepth_[1] = hcons_->getMaxDepth(0);  // HB
  maxDepth_[2] = hcons_->getMaxDepth(1);  // HE
  maxDepth_[3] = hcons_->getMaxDepth(3);  // HO
  maxDepth_[4] = hcons_->getMaxDepth(2);  // HF
  maxDepth_[0] = (maxDepth_[1] > maxDepth_[2] ? maxDepth_[1] : maxDepth_[2]);
  maxDepth_[0] = (maxDepth_[0] > maxDepth_[3] ? maxDepth_[0] : maxDepth_[3]);
  maxDepth_[0] = (maxDepth_[0] > maxDepth_[4] ? maxDepth_[0] : maxDepth_[4]);  // any of HB/HE/HO/HF

  const CaloGeometry* geo = &es.getData(tok_Geom_);
  const HcalGeometry* gHB = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  const HcalGeometry* gHE = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalEndcap));
  const HcalGeometry* gHO = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalOuter));
  const HcalGeometry* gHF = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalForward));

  nChannels_[1] = gHB->getHxSize(1);
  nChannels_[2] = gHE->getHxSize(2);
  nChannels_[3] = gHO->getHxSize(3);
  nChannels_[4] = gHF->getHxSize(4);

  nChannels_[0] = nChannels_[1] + nChannels_[2] + nChannels_[3] + nChannels_[4];

  edm::Service<TFileService> fs;

  // book
  book1D(fs, "nevtot", 1, 0, 1);
  int bnoise = 0;
  int bmc = 0;
  if (subdet_ == "noise")
    bnoise = 1;
  if (mc_ == "yes")
    bmc = 1;
  if (subdet_ == "noise" || subdet_ == "all") {
    booking(fs, "HB", bnoise, bmc);
    booking(fs, "HO", bnoise, bmc);
    booking(fs, "HF", bnoise, bmc);
    booking(fs, "HE", bnoise, bmc);
  } else {
    booking(fs, subdet_, 0, bmc);
  }

  if (skipDataTPs)
    return;

  HistLim tp_hl_et(260, -10, 250);
  HistLim tp_hl_ntp(640, -20, 3180);
  HistLim tp_hl_ntp_sub(404, -20, 2000);
  HistLim tp_hl_ieta(85, -42.5, 42.5);
  HistLim tp_hl_iphi(74, -0.5, 73.5);

  book1D(fs, "HcalDigiTask_tp_et", tp_hl_et);
  book1D(fs, "HcalDigiTask_tp_et_HB", tp_hl_et);
  book1D(fs, "HcalDigiTask_tp_et_HE", tp_hl_et);
  book1D(fs, "HcalDigiTask_tp_et_HF", tp_hl_et);
  book1D(fs, "HcalDigiTask_tp_ntp", tp_hl_ntp);
  book1D(fs, "HcalDigiTask_tp_ntp_HB", tp_hl_ntp_sub);
  book1D(fs, "HcalDigiTask_tp_ntp_HE", tp_hl_ntp_sub);
  book1D(fs, "HcalDigiTask_tp_ntp_HF", tp_hl_ntp_sub);
  book1D(fs, "HcalDigiTask_tp_ntp_ieta", tp_hl_ieta);
  book1D(fs, "HcalDigiTask_tp_ntp_iphi", tp_hl_iphi);
  book1D(fs, "HcalDigiTask_tp_ntp_10_ieta", tp_hl_ieta);
  book2D(fs, "HcalDigiTask_tp_et_ieta", tp_hl_ieta, tp_hl_et);
  book2D(fs, "HcalDigiTask_tp_ieta_iphi", tp_hl_ieta, tp_hl_iphi);
  bookPf(fs, "HcalDigiTask_tp_ave_et_ieta", tp_hl_ieta, tp_hl_et, " ");
  if (Plot_TP_ver_) {
    book1D(fs, "HcalDigiTask_tp_et_v0", tp_hl_et);
    book1D(fs, "HcalDigiTask_tp_et_v1", tp_hl_et);
    book1D(fs, "HcalDigiTask_tp_et_HF_v0", tp_hl_et);
    book1D(fs, "HcalDigiTask_tp_et_HF_v1", tp_hl_et);
    book1D(fs, "HcalDigiTask_tp_ntp_v0", tp_hl_ntp);
    book1D(fs, "HcalDigiTask_tp_ntp_v1", tp_hl_ntp);
    book1D(fs, "HcalDigiTask_tp_ntp_HF_v0", tp_hl_ntp_sub);
    book1D(fs, "HcalDigiTask_tp_ntp_HF_v1", tp_hl_ntp_sub);
    book1D(fs, "HcalDigiTask_tp_ntp_ieta_v0", tp_hl_ieta);
    book1D(fs, "HcalDigiTask_tp_ntp_ieta_v1", tp_hl_ieta);
    book1D(fs, "HcalDigiTask_tp_ntp_iphi_v0", tp_hl_iphi);
    book1D(fs, "HcalDigiTask_tp_ntp_iphi_v1", tp_hl_iphi);
    book1D(fs, "HcalDigiTask_tp_ntp_10_ieta_v0", tp_hl_ieta);
    book1D(fs, "HcalDigiTask_tp_ntp_10_ieta_v1", tp_hl_ieta);
    book2D(fs, "HcalDigiTask_tp_et_ieta_v0", tp_hl_ieta, tp_hl_et);
    book2D(fs, "HcalDigiTask_tp_et_ieta_v1", tp_hl_ieta, tp_hl_et);
    book2D(fs, "HcalDigiTask_tp_ieta_iphi_v0", tp_hl_ieta, tp_hl_iphi);
    book2D(fs, "HcalDigiTask_tp_ieta_iphi_v1", tp_hl_ieta, tp_hl_iphi);
    bookPf(fs, "HcalDigiTask_tp_ave_et_ieta_v0", tp_hl_ieta, tp_hl_et, " ");
    bookPf(fs, "HcalDigiTask_tp_ave_et_ieta_v1", tp_hl_ieta, tp_hl_et, " ");
  }
}

void HcalDigiStudy::booking(edm::Service<TFileService>& fs, const std::string bsubdet, int bnoise, int bmc) {
  // Adjust/Optimize binning (JR Dittmann, 16-JUL-2015)

  HistLim Ndigis(2600, 0., 2600.);
  HistLim sime(200, 0., 1.0);

  HistLim digiAmp(360, -100., 7100.);
  HistLim digiAmpWide(2410, -3000., 720000.);  //300 fC binning
  HistLim ratio(2000, -100., 3900.);
  HistLim sumAmp(100, -500., 1500.);

  HistLim nbin(11, -0.5, 10.5);

  HistLim pedestal(80, -1.0, 15.);
  HistLim pedestalfC(400, -10., 30.);

  HistLim frac(80, -0.20, 1.40);

  HistLim pedLim(80, 0., 8.);
  HistLim pedWidthLim(100, 0., 2.);

  HistLim gainLim(120, 0., 0.6);
  HistLim gainWidthLim(160, 0., 0.32);

  HistLim ietaLim(85, -42.5, 42.5);
  HistLim iphiLim(74, -0.5, 73.5);

  HistLim depthLim(15, -0.5, 14.5);

  //...TDC
  HistLim tdcLim(250, 0., 250.);
  HistLim adcLim(256, 0., 256.);

  if (bsubdet == "HB") {
    Ndigis = HistLim(((int)(nChannels_[1] / 100) + 1) * 100, 0., (float)((int)(nChannels_[1] / 100) + 1) * 100);
  } else if (bsubdet == "HE") {
    sime = HistLim(200, 0., 1.0);
    Ndigis = HistLim(((int)(nChannels_[2] / 100) + 1) * 100, 0., (float)((int)(nChannels_[2] / 100) + 1) * 100);
  } else if (bsubdet == "HF") {
    sime = HistLim(100, 0., 100.);
    pedLim = HistLim(100, 0., 20.);
    pedWidthLim = HistLim(100, 0., 5.);
    frac = HistLim(400, -4.00, 4.00);
    Ndigis = HistLim(((int)(nChannels_[4] / 100) + 1) * 100, 0., (float)((int)(nChannels_[4] / 100) + 1) * 100);
  } else if (bsubdet == "HO") {
    sime = HistLim(200, 0., 1.0);
    gainLim = HistLim(160, 0., 1.6);
    Ndigis = HistLim(((int)(nChannels_[3] / 100) + 1) * 100, 0., (float)((int)(nChannels_[3] / 100) + 1) * 100);
  }

  int isubdet = 0;
  if (bsubdet == "HB")
    isubdet = 1;
  else if (bsubdet == "HE")
    isubdet = 2;
  else if (bsubdet == "HO")
    isubdet = 3;
  else if (bsubdet == "HF")
    isubdet = 4;
  else
    edm::LogWarning("HcalDigiStudy") << "HcalDigiStudy Warning: not HB/HE/HF/HO " << bsubdet << std::endl;

  Char_t histo[100];
  const char* sub = bsubdet.c_str();
  if (bnoise == 0) {
    // number of digis in each subdetector
    sprintf(histo, "HcalDigiTask_Ndigis_%s", sub);
    book1D(fs, histo, Ndigis);

    // maps of occupancies
    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_ieta_iphi_occupancy_map_depth%d_%s", depth, sub);
      book2D(fs, histo, ietaLim, iphiLim);
    }

    //Depths
    sprintf(histo, "HcalDigiTask_depths_%s", sub);
    book1D(fs, histo, depthLim);

    // occupancies vs ieta
    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_occupancy_vs_ieta_depth%d_%s", depth, sub);
      book1D(fs, histo, ietaLim);
    }

    // just 1D of all cells' amplitudes
    sprintf(histo, "HcalDigiTask_sum_all_amplitudes_%s", sub);
    if ((HBPhase1_ && bsubdet == "HB") || (HEPhase1_ && bsubdet == "HE"))
      book1D(fs, histo, digiAmpWide);
    else
      book1D(fs, histo, digiAmp);

    sprintf(histo, "HcalDigiTask_number_of_amplitudes_above_10fC_%s", sub);
    book1D(fs, histo, Ndigis);

    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_ADC0_adc_depth%d_%s", depth, sub);
      book1D(fs, histo, pedestal);
    }

    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_ADC0_fC_depth%d_%s", depth, sub);
      book1D(fs, histo, pedestalfC);
    }

    sprintf(histo, "HcalDigiTask_signal_amplitude_%s", sub);
    if ((HBPhase1_ && bsubdet == "HB") || (HEPhase1_ && bsubdet == "HE"))
      book1D(fs, histo, digiAmpWide);
    else
      book1D(fs, histo, digiAmp);

    if (hep17_ && bsubdet == "HE") {
      sprintf(histo, "HcalDigiTask_signal_amplitude_HEP17");
      book1D(fs, histo, digiAmpWide);
    }
    //
    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_signal_amplitude_depth%d_%s", depth, sub);
      if ((HBPhase1_ && bsubdet == "HB") || (HEPhase1_ && bsubdet == "HE"))
        book1D(fs, histo, digiAmpWide);
      else
        book1D(fs, histo, digiAmp);
      if (hep17_ && bsubdet == "HE") {
        sprintf(histo, "HcalDigiTask_signal_amplitude_depth%d_HEP17", depth);
        book1D(fs, histo, digiAmpWide);
      }
    }

    sprintf(histo, "HcalDigiTask_signal_amplitude_vs_bin_all_depths_%s", sub);
    if ((HBPhase1_ && bsubdet == "HB") || (HEPhase1_ && bsubdet == "HE"))
      book2D(fs, histo, nbin, digiAmpWide);
    else
      book2D(fs, histo, nbin, digiAmp);
    if (hep17_ && bsubdet == "HE") {
      sprintf(histo, "HcalDigiTask_signal_amplitude_vs_bin_all_depths_HEP17");
      book2D(fs, histo, nbin, digiAmpWide);
    }

    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_all_amplitudes_vs_bin_1D_depth%d_%s", depth, sub);
      book1D(fs, histo, nbin);
      if (hep17_ && bsubdet == "HE") {
        sprintf(histo, "HcalDigiTask_all_amplitudes_vs_bin_1D_depth%d_HEP17", depth);
        book1D(fs, histo, nbin);
      }
    }

    sprintf(histo, "HcalDigiTask_SOI_frac_%s", sub);
    book1D(fs, histo, frac);
    sprintf(histo, "HcalDigiTask_postSOI_frac_%s", sub);
    book1D(fs, histo, frac);

    if (bmc == 1) {
      sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_%s", sub);
      book2D(fs, histo, sime, digiAmp);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_depth%d_%s", depth, sub);
        book2D(fs, histo, sime, digiAmp);
      }

      sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_profile_%s", sub);
      bookPf(fs, histo, sime, digiAmp);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        sprintf(histo, "HcalDigiTask_amplitude_vs_simhits_profile_depth%d_%s", depth, sub);
        bookPf(fs, histo, sime, digiAmp);
      }

      sprintf(histo, "HcalDigiTask_ratio_amplitude_vs_simhits_%s", sub);
      book1D(fs, histo, ratio);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        sprintf(histo, "HcalDigiTask_ratio_amplitude_vs_simhits_depth%d_%s", depth, sub);
        book1D(fs, histo, ratio);
      }

      //...TDC
      if (bsubdet == "HB" || bsubdet == "HE") {
        sprintf(histo, "HcalDigiTask_TDCtime_%s", sub);
        book1D(fs, histo, tdcLim);

        sprintf(histo, "HcalDigiTask_TDCtime_vs_ADC_%s", sub);
        book2D(fs, histo, adcLim, tdcLim);
      }

    }  //mc only

  } else {  // noise only

    // EVENT "1" distributions of all cells properties

    //KH
    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_gain_capId0_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainLim);
      sprintf(histo, "HcalDigiTask_gain_capId1_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainLim);
      sprintf(histo, "HcalDigiTask_gain_capId2_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainLim);
      sprintf(histo, "HcalDigiTask_gain_capId3_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainLim);

      sprintf(histo, "HcalDigiTask_gainWidth_capId0_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainWidthLim);
      sprintf(histo, "HcalDigiTask_gainWidth_capId1_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainWidthLim);
      sprintf(histo, "HcalDigiTask_gainWidth_capId2_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainWidthLim);
      sprintf(histo, "HcalDigiTask_gainWidth_capId3_Depth%d_%s", depth, sub);
      book1D(fs, histo, gainWidthLim);

      sprintf(histo, "HcalDigiTask_pedestal_capId0_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedLim);
      sprintf(histo, "HcalDigiTask_pedestal_capId1_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedLim);
      sprintf(histo, "HcalDigiTask_pedestal_capId2_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedLim);
      sprintf(histo, "HcalDigiTask_pedestal_capId3_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedLim);

      sprintf(histo, "HcalDigiTask_pedestal_width_capId0_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedWidthLim);
      sprintf(histo, "HcalDigiTask_pedestal_width_capId1_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedWidthLim);
      sprintf(histo, "HcalDigiTask_pedestal_width_capId2_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedWidthLim);
      sprintf(histo, "HcalDigiTask_pedestal_width_capId3_Depth%d_%s", depth, sub);
      book1D(fs, histo, pedWidthLim);
    }

    //KH
    for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
      sprintf(histo, "HcalDigiTask_gainMap_Depth%d_%s", depth, sub);
      book2D(fs, histo, ietaLim, iphiLim);
      sprintf(histo, "HcalDigiTask_pwidthMap_Depth%d_%s", depth, sub);
      book2D(fs, histo, ietaLim, iphiLim);
    }

  }  //end of noise-only
}  //book

void HcalDigiStudy::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  conditions_ = &iSetup.getData(tok_Cond_);

  //TP Code
  const auto& decoder = (!skipTPs) ? &iSetup.getData(tok_Decoder_) : nullptr;
  const auto& tp_geometry = (!skipTPs) ? &iSetup.getData(tok_TPGeom_) : nullptr;
  htopo_ = &iSetup.getData(tok_Topo_);

  //Get all handles
  edm::Handle<HcalTrigPrimDigiCollection> emulTPs;
  iEvent.getByToken(tok_emulTPs_, emulTPs);

  edm::Handle<HcalTrigPrimDigiCollection> dataTPs;
  if (!skipDataTPs)
    iEvent.getByToken(tok_dataTPs_, dataTPs);
  //iEvent.getByLabel("hcalDigis", dataTPs);

  //~TP Code

  if (subdet_ != "all") {
    noise_ = 0;
    if (subdet_ == "HB") {
      reco<HBHEDataFrame>(iEvent, iSetup, tok_hbhe_);
      reco<QIE11DataFrame>(iEvent, iSetup, tok_qie11_hbhe_);
    }
    if (subdet_ == "HE") {
      reco<HBHEDataFrame>(iEvent, iSetup, tok_hbhe_);
      reco<QIE11DataFrame>(iEvent, iSetup, tok_qie11_hbhe_);
    }
    if (subdet_ == "HO")
      reco<HODataFrame>(iEvent, iSetup, tok_ho_);
    if (subdet_ == "HF") {
      reco<HFDataFrame>(iEvent, iSetup, tok_hf_);
      reco<QIE10DataFrame>(iEvent, iSetup, tok_qie10_hf_);
    }

    if (subdet_ == "noise") {
      noise_ = 1;
      subdet_ = "HB";
      reco<HBHEDataFrame>(iEvent, iSetup, tok_hbhe_);
      reco<QIE11DataFrame>(iEvent, iSetup, tok_qie11_hbhe_);
      subdet_ = "HE";
      reco<HBHEDataFrame>(iEvent, iSetup, tok_hbhe_);
      reco<QIE11DataFrame>(iEvent, iSetup, tok_qie11_hbhe_);
      subdet_ = "HO";
      reco<HODataFrame>(iEvent, iSetup, tok_ho_);
      subdet_ = "HF";
      reco<HFDataFrame>(iEvent, iSetup, tok_hf_);
      reco<QIE10DataFrame>(iEvent, iSetup, tok_qie10_hf_);
      subdet_ = "noise";
    }
  }  // all subdetectors
  else {
    noise_ = 0;

    subdet_ = "HB";
    reco<HBHEDataFrame>(iEvent, iSetup, tok_hbhe_);
    reco<QIE11DataFrame>(iEvent, iSetup, tok_qie11_hbhe_);
    subdet_ = "HE";
    reco<HBHEDataFrame>(iEvent, iSetup, tok_hbhe_);
    reco<QIE11DataFrame>(iEvent, iSetup, tok_qie11_hbhe_);
    subdet_ = "HO";
    reco<HODataFrame>(iEvent, iSetup, tok_ho_);
    subdet_ = "HF";
    reco<HFDataFrame>(iEvent, iSetup, tok_hf_);
    reco<QIE10DataFrame>(iEvent, iSetup, tok_qie10_hf_);
    subdet_ = "all";
  }

  fill1D("nevtot", 0);
  nevtot++;

  //TP Code
  //Counters
  int c = 0, chb = 0, che = 0, chf = 0, cv0 = 0, cv1 = 0, chfv0 = 0, chfv1 = 0;

  if (skipDataTPs)
    return;

  for (HcalTrigPrimDigiCollection::const_iterator itr = dataTPs->begin(); itr != dataTPs->end(); ++itr) {
    int ieta = itr->id().ieta();
    int iphi = itr->id().iphi();

    HcalSubdetector subdet = (HcalSubdetector)0;

    if (abs(ieta) <= 16)
      subdet = HcalSubdetector::HcalBarrel;
    else if (abs(ieta) < tp_geometry->firstHFTower(itr->id().version()))
      subdet = HcalSubdetector::HcalEndcap;
    else if (abs(ieta) <= 42)
      subdet = HcalSubdetector::HcalForward;

    //Right now, the only case where version matters is in HF
    //If the subdetector is not HF, set version to -1
    int tpVersion = (subdet == HcalSubdetector::HcalForward ? itr->id().version() : -1);

    float en = decoder->hcaletValue(itr->id(), itr->t0());

    if (en < 0.00001)
      continue;

    //Plot the variables
    //Retain classic behavior (include all tps)
    //Additional plots that only include HF 3x2 or HF 1x1

    //Classics
    fill1D("HcalDigiTask_tp_et", en);
    fill2D("HcalDigiTask_tp_et_ieta", ieta, en);
    fill2D("HcalDigiTask_tp_ieta_iphi", ieta, iphi);
    fillPf("HcalDigiTask_tp_ave_et_ieta", ieta, en);
    fill1D("HcalDigiTask_tp_ntp_ieta", ieta);
    fill1D("HcalDigiTask_tp_ntp_iphi", iphi);
    if (en > 10.)
      fill1D("HcalDigiTask_tp_ntp_10_ieta", ieta);

    //3x2 Trig Primitives (tpVersion == 0)
    if ((subdet != HcalSubdetector::HcalForward || tpVersion == 0) && Plot_TP_ver_) {
      fill1D("HcalDigiTask_tp_et_v0", en);
      fill2D("HcalDigiTask_tp_et_ieta_v0", ieta, en);
      fill2D("HcalDigiTask_tp_ieta_iphi_v0", ieta, iphi);
      fillPf("HcalDigiTask_tp_ave_et_ieta_v0", ieta, en);
      fill1D("HcalDigiTask_tp_ntp_ieta_v0", ieta);
      fill1D("HcalDigiTask_tp_ntp_iphi_v0", iphi);
      if (en > 10.)
        fill1D("HcalDigiTask_tp_ntp_10_ieta_v0", ieta);
    }

    //1x1 Trig Primitives (tpVersion == 1)
    if ((subdet != HcalSubdetector::HcalForward || tpVersion == 1) && Plot_TP_ver_) {
      fill1D("HcalDigiTask_tp_et_v1", en);
      fill2D("HcalDigiTask_tp_et_ieta_v1", ieta, en);
      fill2D("HcalDigiTask_tp_ieta_iphi_v1", ieta, iphi);
      fillPf("HcalDigiTask_tp_ave_et_ieta_v1", ieta, en);
      fill1D("HcalDigiTask_tp_ntp_ieta_v1", ieta);
      fill1D("HcalDigiTask_tp_ntp_iphi_v1", iphi);
      if (en > 10.)
        fill1D("HcalDigiTask_tp_ntp_10_ieta_v1", ieta);
    }

    ++c;
    if (subdet == HcalSubdetector::HcalBarrel) {
      fill1D("HcalDigiTask_tp_et_HB", en);
      ++chb;
      if (Plot_TP_ver_) {
        ++cv0;
        ++cv1;
      }
    }
    if (subdet == HcalSubdetector::HcalEndcap) {
      fill1D("HcalDigiTask_tp_et_HE", en);
      ++che;
      if (Plot_TP_ver_) {
        ++cv0;
        ++cv1;
      }
    }
    if (subdet == HcalSubdetector::HcalForward) {
      fill1D("HcalDigiTask_tp_et_HF", en);
      ++chf;

      if (tpVersion == 0 && Plot_TP_ver_) {
        fill1D("HcalDigiTask_tp_et_HF_v0", en);
        ++chfv0;
        ++cv0;
      }

      if (tpVersion == 1 && Plot_TP_ver_) {
        fill1D("HcalDigiTask_tp_et_HF_v1", en);
        ++chfv1;
        ++cv1;
      }
    }

  }  //end data TP collection

  fill1D("HcalDigiTask_tp_ntp", c);
  fill1D("HcalDigiTask_tp_ntp_HB", chb);
  fill1D("HcalDigiTask_tp_ntp_HE", che);
  fill1D("HcalDigiTask_tp_ntp_HF", chf);
  if (Plot_TP_ver_) {
    fill1D("HcalDigiTask_tp_ntp_v0", cv0);
    fill1D("HcalDigiTask_tp_ntp_v1", cv1);
    fill1D("HcalDigiTask_tp_ntp_HF_v0", chfv0);
    fill1D("HcalDigiTask_tp_ntp_HF_v1", chfv1);
  }

  //~TP Code
}

template <class Digi>
void HcalDigiStudy::reco(const edm::Event& iEvent,
                         const edm::EventSetup& iSetup,
                         const edm::EDGetTokenT<edm::SortedCollection<Digi> >& tok) {
  // HistLim =============================================================

  std::string strtmp;

  // ======================================================================
  typename edm::Handle<edm::SortedCollection<Digi> > digiCollection;
  typename edm::SortedCollection<Digi>::const_iterator digiItr;

  // ADC2fC
  CaloSamples tool;
  iEvent.getByToken(tok, digiCollection);
  if (!digiCollection.isValid())
    return;
  int isubdet = 0;
  if (subdet_ == "HB")
    isubdet = 1;
  if (subdet_ == "HE")
    isubdet = 2;
  if (subdet_ == "HO")
    isubdet = 3;
  if (subdet_ == "HF")
    isubdet = 4;

  if (isubdet == 1)
    nevent1++;
  if (isubdet == 2)
    nevent2++;
  if (isubdet == 3)
    nevent3++;
  if (isubdet == 4)
    nevent4++;

  int indigis = 0;
  //  amplitude for signal cell at diff. depths
  std::vector<double> v_ampl_c(maxDepth_[isubdet] + 1, 0);

  // is set to 1 if "seed" SimHit is found
  int seedSimHit = 0;

  int ieta_Sim = 9999;
  int iphi_Sim = 9999;
  double emax_Sim = -9999.;

  // SimHits MC only
  if (mc_ == "yes") {
    edm::Handle<edm::PCaloHitContainer> hcalHits;
    iEvent.getByToken(tok_mc_, hcalHits);
    const edm::PCaloHitContainer* simhitResult = hcalHits.product();

    if (isubdet != 0 && noise_ == 0) {  // signal only SimHits

      for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin(); simhits != simhitResult->end();
           ++simhits) {
        unsigned int id = simhits->id();
        int sub, ieta, iphi;
        HcalDetId hid;
        if (testNumber_)
          hid = HcalHitRelabeller::relabel(id, hcons_);
        else
          hid = HcalDetId(id);
        sub = hid.subdet();
        ieta = hid.ieta();
        iphi = hid.iphi();

        double en = simhits->energy();

        if (en > emax_Sim && sub == isubdet) {
          emax_Sim = en;
          ieta_Sim = ieta;
          iphi_Sim = iphi;
          // to limit "seed" SimHit energy in case of "multi" event
          if (mode_ == "multi" && ((sub == 4 && en < 100. && en > 1.) || ((sub != 4) && en < 1. && en > 0.02))) {
            seedSimHit = 1;
            break;
          }
        }

      }  // end of SimHits cycle

      // found highest-energy SimHit for single-particle
      if (mode_ != "multi" && emax_Sim > 0.)
        seedSimHit = 1;
    }  // end of SimHits
  }  // end of mc_ == "yes"

  // CYCLE OVER CELLS ========================================================
  int Ndig = 0;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalDiguStudy") << "Subdet " << subdet_ << " with " << digiCollection->size()
                                    << " entries in DigiCollection";
#endif
  for (digiItr = digiCollection->begin(); digiItr != digiCollection->end(); digiItr++) {
    HcalDetId cell(digiItr->id());
    int depth = cell.depth();
    int iphi = cell.iphi();
    int ieta = cell.ieta();
    int sub = cell.subdet();

    if (depth > maxDepth_[isubdet] && sub == isubdet) {
      edm::LogWarning("HcalDetId") << "HcalDetID presents conflicting information. Depth: " << depth
                                   << ", iphi: " << iphi << ", ieta: " << ieta
                                   << ". Max depth from geometry is: " << maxDepth_[isubdet]
                                   << ". TestNumber = " << testNumber_;
      continue;
    }

    //  amplitude for signal cell at diff. depths
    std::vector<double> v_ampl(maxDepth_[isubdet] + 1, 0);

    // Gains, pedestals (once !) and only for "noise" case
    if (((nevent1 == 1 && isubdet == 1) || (nevent2 == 1 && isubdet == 2) || (nevent3 == 1 && isubdet == 3) ||
         (nevent4 == 1 && isubdet == 4)) &&
        noise_ == 1 && sub == isubdet) {
      HcalGenericDetId hcalGenDetId(digiItr->id());
      const HcalPedestal* pedestal = conditions_->getPedestal(hcalGenDetId);
      const HcalGain* gain = conditions_->getGain(hcalGenDetId);
      const HcalGainWidth* gainWidth = conditions_->getGainWidth(hcalGenDetId);
      const HcalPedestalWidth* pedWidth = conditions_->getPedestalWidth(hcalGenDetId);

      for (int i = 0; i < 4; i++) {
        fill1D("HcalDigiTask_gain_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, gain->getValue(i));
        fill1D("HcalDigiTask_gainWidth_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, gainWidth->getValue(i));
        fill1D("HcalDigiTask_pedestal_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, pedestal->getValue(i));
        fill1D("HcalDigiTask_pedestal_width_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_,
               pedWidth->getWidth(i));
      }

      fill2D("HcalDigiTask_gainMap_Depth" + str(depth) + "_" + subdet_, double(ieta), double(iphi), gain->getValue(0));
      fill2D("HcalDigiTask_pwidthMap_Depth" + str(depth) + "_" + subdet_,
             double(ieta),
             double(iphi),
             pedWidth->getWidth(0));

    }  // end of event #1

    if (sub == isubdet)
      Ndig++;  // subdet number of digi

    // No-noise case, only single  subdet selected  ===========================

    if (sub == isubdet && noise_ == 0) {
      HcalCalibrations calibrations = conditions_->getHcalCalibrations(cell);

      const HcalQIECoder* channelCoder = conditions_->getHcalCoder(cell);
      const HcalQIEShape* shape = conditions_->getHcalShape(channelCoder);
      HcalCoderDb coder(*channelCoder, *shape);
      coder.adc2fC(*digiItr, tool);

      // for dynamic digi time sample analysis
      int soi = tool.presamples();
      int lastbin = tool.size() - 1;

      double noiseADC = (*digiItr)[0].adc();
      double noisefC = tool[0];
      // noise evaluations from "pre-samples"
      fill1D("HcalDigiTask_ADC0_adc_depth" + str(depth) + "_" + subdet_, noiseADC);
      fill1D("HcalDigiTask_ADC0_fC_depth" + str(depth) + "_" + subdet_, noisefC);

      // OCCUPANCY maps fill
      fill2D("HcalDigiTask_ieta_iphi_occupancy_map_depth" + str(depth) + "_" + subdet_, double(ieta), double(iphi));

      fill1D("HcalDigiTask_depths_" + subdet_, double(depth));

      // Cycle on time slices
      // - for each Digi
      // - for one Digi with max SimHits E in subdet

      int closen = 0;  // =1 if 1) seedSimHit = 1 and 2) the cell is the same
      if (ieta == ieta_Sim && iphi == iphi_Sim)
        closen = seedSimHit;

      for (int ii = 0; ii < tool.size(); ii++) {
        int capid = (*digiItr)[ii].capid();
        // single ts amplitude
        double val = (tool[ii] - calibrations.pedestal(capid));

        if (val > 100.) {
          fill1D("HcalDigiTask_ADC0_adc_depth" + str(depth) + "_" + subdet_, noiseADC);
          strtmp = "HcalDigiTask_all_amplitudes_vs_bin_1D_depth" + str(depth) + "_" + subdet_;
          fill1D(strtmp, double(ii), val);
        }

        if (closen == 1) {
          strtmp = "HcalDigiTask_signal_amplitude_vs_bin_all_depths_" + subdet_;
          fill2D(strtmp, double(ii), val);
        }

        // all detectors
        if (ii >= soi && ii <= lastbin) {
          v_ampl[0] += val;
          v_ampl[depth] += val;

          if (closen == 1) {
            v_ampl_c[0] += val;
            v_ampl_c[depth] += val;
          }
        }
      }
      // end of time bucket sample

      // maps of sum of amplitudes (sum lin.digis(4,5,6,7) - ped) all depths
      // just 1D of all cells' amplitudes
      strtmp = "HcalDigiTask_sum_all_amplitudes_" + subdet_;
      fill1D(strtmp, v_ampl[0]);

      std::vector<int> v_ampl_sub(v_ampl.begin() + 1, v_ampl.end());  // remove element 0, which is the sum of any depth
      double ampl_max = *std::max_element(v_ampl_sub.begin(), v_ampl_sub.end());
      if (ampl_max > 10.)
        indigis++;
      //KH if (ampl1 > 10. || ampl2 > 10. || ampl3 > 10. || ampl4 > 10.) indigis++;

      // fraction 5,6 bins if ampl. is big.
      if (v_ampl[depth] > 30.) {
        double fbinSOI = tool[soi] - calibrations.pedestal((*digiItr)[soi].capid());
        double fbinPS = 0;

        for (int j = soi + 1; j <= lastbin; j++)
          fbinPS += tool[j] - calibrations.pedestal((*digiItr)[j].capid());

        fbinSOI /= v_ampl[depth];
        fbinPS /= v_ampl[depth];
        strtmp = "HcalDigiTask_SOI_frac_" + subdet_;
        fill1D(strtmp, fbinSOI);
        strtmp = "HcalDigiTask_postSOI_frac_" + subdet_;
        fill1D(strtmp, fbinPS);
      }

      strtmp = "HcalDigiTask_signal_amplitude_" + subdet_;
      fill1D(strtmp, v_ampl[0]);
      strtmp = "HcalDigiTask_signal_amplitude_depth" + str(depth) + "_" + subdet_;
      fill1D(strtmp, v_ampl[depth]);
    }
  }  // End of CYCLE OVER CELLS =============================================

  if (isubdet != 0 && noise_ == 0) {  // signal only, once per event
    strtmp = "HcalDigiTask_number_of_amplitudes_above_10fC_" + subdet_;
    fill1D(strtmp, indigis);

    // SimHits once again !!!
    double eps = 1.e-3;
    std::vector<double> v_ehits(maxDepth_[isubdet] + 1, 0);

    if (mc_ == "yes") {
      edm::Handle<edm::PCaloHitContainer> hcalHits;
      iEvent.getByToken(tok_mc_, hcalHits);
      const edm::PCaloHitContainer* simhitResult = hcalHits.product();
      for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin(); simhits != simhitResult->end();
           ++simhits) {
        unsigned int id = simhits->id();
        int sub, depth, ieta, iphi;
        HcalDetId hid;
        if (testNumber_)
          hid = HcalHitRelabeller::relabel(id, hcons_);
        else
          hid = HcalDetId(id);
        sub = hid.subdet();
        depth = hid.depth();
        ieta = hid.ieta();
        iphi = hid.iphi();

        if (depth > maxDepth_[isubdet] && sub == isubdet) {
          edm::LogWarning("HcalDetId") << "HcalDetID(SimHit) presents conflicting information. Depth: " << depth
                                       << ", iphi: " << iphi << ", ieta: " << ieta
                                       << ". Max depth from geometry is: " << maxDepth_[isubdet]
                                       << ". TestNumber = " << testNumber_;
          continue;
        }

        // take cell already found to be max energy in a particular subdet
        if (sub == isubdet && ieta == ieta_Sim && iphi == iphi_Sim) {
          double en = simhits->energy();

          v_ehits[0] += en;
          v_ehits[depth] += en;
        }
      }  // simhit loop

      strtmp = "HcalDigiTask_amplitude_vs_simhits_" + subdet_;
      if (v_ehits[0] > eps)
        fill2D(strtmp, v_ehits[0], v_ampl_c[0]);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        strtmp = "HcalDigiTask_amplitude_vs_simhits_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fill2D(strtmp, v_ehits[depth], v_ampl_c[depth]);
      }

      strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_" + subdet_;
      if (v_ehits[0] > eps)
        fillPf(strtmp, v_ehits[0], v_ampl_c[0]);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fillPf(strtmp, v_ehits[depth], v_ampl_c[depth]);
      }

      strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_" + subdet_;
      if (v_ehits[0] > eps)
        fill1D(strtmp, v_ampl_c[0] / v_ehits[0]);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fillPf(strtmp, v_ehits[depth], v_ampl_c[depth]);
        strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fill1D(strtmp, v_ampl_c[depth] / v_ehits[depth]);
      }

    }  // end of if(mc_ == "yes")

    strtmp = "HcalDigiTask_Ndigis_" + subdet_;
    fill1D(strtmp, double(Ndig));

  }  //  end of if( subdet != 0 && noise_ == 0) { // signal only
}
template <class dataFrameType>
void HcalDigiStudy::reco(const edm::Event& iEvent,
                         const edm::EventSetup& iSetup,
                         const edm::EDGetTokenT<HcalDataFrameContainer<dataFrameType> >& tok) {
  // HistLim =============================================================

  std::string strtmp;

  // ======================================================================
  typename edm::Handle<HcalDataFrameContainer<dataFrameType> > digiHandle;
  //typename HcalDataFrameContainer<dataFrameType>::const_iterator digiItr;

  // ADC2fC
  CaloSamples tool;
  iEvent.getByToken(tok, digiHandle);
  if (!digiHandle.isValid())
    return;
  const HcalDataFrameContainer<dataFrameType>* digiCollection = digiHandle.product();
  int isubdet = 0;
  if (subdet_ == "HB")
    isubdet = 1;
  if (subdet_ == "HE")
    isubdet = 2;
  if (subdet_ == "HO")
    isubdet = 3;
  if (subdet_ == "HF")
    isubdet = 4;

  if (isubdet == 1)
    nevent1++;
  if (isubdet == 2)
    nevent2++;
  if (isubdet == 3)
    nevent3++;
  if (isubdet == 4)
    nevent4++;

  int indigis = 0;
  //  amplitude for signal cell at diff. depths
  std::vector<double> v_ampl_c(maxDepth_[isubdet] + 1, 0);

  // is set to 1 if "seed" SimHit is found
  int seedSimHit = 0;

  int ieta_Sim = 9999;
  int iphi_Sim = 9999;
  double emax_Sim = -9999.;

  // SimHits MC only
  if (mc_ == "yes") {
    edm::Handle<edm::PCaloHitContainer> hcalHits;
    iEvent.getByToken(tok_mc_, hcalHits);
    const edm::PCaloHitContainer* simhitResult = hcalHits.product();

    if (isubdet != 0 && noise_ == 0) {  // signal only SimHits

      for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin(); simhits != simhitResult->end();
           ++simhits) {
        unsigned int id = simhits->id();
        int sub, ieta, iphi;
        HcalDetId hid;
        if (testNumber_)
          hid = HcalHitRelabeller::relabel(id, hcons_);
        else
          hid = HcalDetId(id);
        sub = hid.subdet();
        ieta = hid.ieta();
        iphi = hid.iphi();

        double en = simhits->energy();

        if (en > emax_Sim && sub == isubdet) {
          emax_Sim = en;
          ieta_Sim = ieta;
          iphi_Sim = iphi;
          // to limit "seed" SimHit energy in case of "multi" event
          if (mode_ == "multi" && ((sub == 4 && en < 100. && en > 1.) || ((sub != 4) && en < 1. && en > 0.02))) {
            seedSimHit = 1;
            break;
          }
        }

      }  // end of SimHits cycle

      // found highest-energy SimHit for single-particle
      if (mode_ != "multi" && emax_Sim > 0.)
        seedSimHit = 1;
    }  // end of SimHits
  }  // end of mc_ == "yes"

  // CYCLE OVER CELLS ========================================================
  int Ndig = 0;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalDiguStudy") << "Subdet " << subdet_ << " with " << digiCollection->size()
                                    << " entries in DigiCollection";
#endif

  for (typename HcalDataFrameContainer<dataFrameType>::const_iterator digiItr = digiCollection->begin();
       digiItr != digiCollection->end();
       digiItr++) {
    dataFrameType dataFrame = *digiItr;

    HcalDetId cell(digiItr->id());
    int depth = cell.depth();
    int iphi = cell.iphi();
    int ieta = cell.ieta();
    int sub = cell.subdet();

    //Is this in HEP17
    bool isHEP17 = (iphi >= 63) && (iphi <= 66) && (ieta > 0) && (sub == 2);

    if (depth > maxDepth_[isubdet] && sub == isubdet) {
      edm::LogWarning("HcalDetId") << "HcalDetID presents conflicting information. Depth: " << depth
                                   << ", iphi: " << iphi << ", ieta: " << ieta
                                   << ". Max depth from geometry is: " << maxDepth_[isubdet]
                                   << ". TestNumber = " << testNumber_;
      continue;
    }

    //  amplitude for signal cell at diff. depths
    std::vector<double> v_ampl(maxDepth_[isubdet] + 1, 0);

    // Gains, pedestals (once !) and only for "noise" case
    if (((nevent1 == 1 && isubdet == 1) || (nevent2 == 1 && isubdet == 2) || (nevent3 == 1 && isubdet == 3) ||
         (nevent4 == 1 && isubdet == 4)) &&
        noise_ == 1 && sub == isubdet) {
      HcalGenericDetId hcalGenDetId(digiItr->id());
      const HcalPedestal* pedestal = conditions_->getPedestal(hcalGenDetId);
      const HcalGain* gain = conditions_->getGain(hcalGenDetId);
      const HcalGainWidth* gainWidth = conditions_->getGainWidth(hcalGenDetId);
      const HcalPedestalWidth* pedWidth = conditions_->getPedestalWidth(hcalGenDetId);

      for (int i = 0; i < 4; i++) {
        fill1D("HcalDigiTask_gain_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, gain->getValue(i));
        fill1D("HcalDigiTask_gainWidth_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, gainWidth->getValue(i));
        fill1D("HcalDigiTask_pedestal_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_, pedestal->getValue(i));
        fill1D("HcalDigiTask_pedestal_width_capId" + str(i) + "_Depth" + str(depth) + "_" + subdet_,
               pedWidth->getWidth(i));
      }

      fill2D("HcalDigiTask_gainMap_Depth" + str(depth) + "_" + subdet_, double(ieta), double(iphi), gain->getValue(0));
      fill2D("HcalDigiTask_pwidthMap_Depth" + str(depth) + "_" + subdet_,
             double(ieta),
             double(iphi),
             pedWidth->getWidth(0));

    }  // end of event #1
    //#ifdef EDM_ML_DEBUG
    //    edm::LogVerbatim("HcalDigiStudy") << "==== End of event noise block in cell cycle";
    //#endif
    if (sub == isubdet)
      Ndig++;  // subdet number of digi

    // No-noise case, only single  subdet selected  ===========================

    if (sub == isubdet && noise_ == 0) {
      HcalCalibrations calibrations = conditions_->getHcalCalibrations(cell);

      const HcalQIECoder* channelCoder = conditions_->getHcalCoder(cell);
      const HcalQIEShape* shape = conditions_->getHcalShape(channelCoder);
      HcalCoderDb coder(*channelCoder, *shape);
      coder.adc2fC(dataFrame, tool);

      // for dynamic digi time sample analysis
      int soi = tool.presamples();
      int lastbin = tool.size() - 1;

      double noiseADC = (dataFrame)[0].adc();
      double noisefC = tool[0];
      // noise evaluations from "pre-samples"
      fill1D("HcalDigiTask_ADC0_adc_depth" + str(depth) + "_" + subdet_, noiseADC);
      fill1D("HcalDigiTask_ADC0_fC_depth" + str(depth) + "_" + subdet_, noisefC);

      // OCCUPANCY maps fill
      fill2D("HcalDigiTask_ieta_iphi_occupancy_map_depth" + str(depth) + "_" + subdet_, double(ieta), double(iphi));

      fill1D("HcalDigiTask_depths_" + subdet_, double(depth));

      // Cycle on time slices
      // - for each Digi
      // - for one Digi with max SimHits E in subdet

      int closen = 0;  // =1 if 1) seedSimHit = 1 and 2) the cell is the same
      if (ieta == ieta_Sim && iphi == iphi_Sim)
        closen = seedSimHit;

      for (int ii = 0; ii < tool.size(); ii++) {
        int capid = (dataFrame)[ii].capid();
        // single ts amplitude
        double val = (tool[ii] - calibrations.pedestal(capid));

        if (val > 100.) {
          fill1D("HcalDigiTask_ADC0_adc_depth" + str(depth) + "_" + subdet_, noiseADC);
          if (hep17_) {
            if (!isHEP17) {
              strtmp = "HcalDigiTask_all_amplitudes_vs_bin_1D_depth" + str(depth) + "_" + subdet_;
              fill1D(strtmp, double(ii), val);
            } else {
              strtmp = "HcalDigiTask_all_amplitudes_vs_bin_1D_depth" + str(depth) + "_HEP17";
              fill1D(strtmp, double(ii), val);
            }
          } else {
            strtmp = "HcalDigiTask_all_amplitudes_vs_bin_1D_depth" + str(depth) + "_" + subdet_;
            fill1D(strtmp, double(ii), val);
          }
        }

        if (closen == 1) {
          if (hep17_) {
            if (!isHEP17) {
              strtmp = "HcalDigiTask_signal_amplitude_vs_bin_all_depths_" + subdet_;
              fill2D(strtmp, double(ii), val);
            } else {
              strtmp = "HcalDigiTask_signal_amplitude_vs_bin_all_depths_HEP17";
              fill2D(strtmp, double(ii), val);
            }
          } else {
            strtmp = "HcalDigiTask_signal_amplitude_vs_bin_all_depths_" + subdet_;
            fill2D(strtmp, double(ii), val);
          }
        }

        // all detectors
        if (ii >= soi && ii <= lastbin) {
          v_ampl[0] += val;
          v_ampl[depth] += val;

          if (closen == 1) {
            v_ampl_c[0] += val;
            v_ampl_c[depth] += val;
          }
        }

        //...TDC

        if ((HBPhase1_ && sub == 1) || (HEPhase1_ && sub == 2)) {
          double digiADC = (dataFrame)[ii].adc();
          const QIE11DataFrame dataFrameHBHE = static_cast<const QIE11DataFrame>(*digiItr);
          double digiTDC = (dataFrameHBHE)[ii].tdc();
          if (digiTDC < 50) {
            double time = ii * 25. + (digiTDC * 0.5);
            strtmp = "HcalDigiTask_TDCtime_" + subdet_;
            fill1D(strtmp, time);

            strtmp = "HcalDigiTask_TDCtime_vs_ADC_" + subdet_;
            fill2D(strtmp, digiADC, time);
          }
        }
      }
      // end of time bucket sample

      // just 1D of all cells' amplitudes
      strtmp = "HcalDigiTask_sum_all_amplitudes_" + subdet_;
      fill1D(strtmp, v_ampl[0]);

      std::vector<int> v_ampl_sub(v_ampl.begin() + 1, v_ampl.end());  // remove element 0, which is the sum of any depth
      double ampl_max = *std::max_element(v_ampl_sub.begin(), v_ampl_sub.end());
      if (ampl_max > 10.)
        indigis++;
      //KH if (ampl1 > 10. || ampl2 > 10. || ampl3 > 10. || ampl4 > 10.) indigis++;

      // fraction 5,6 bins if ampl. is big.
      //histogram names have not been changed, but it should be understood that bin_5 is soi, and bin_6_7 is latter TS'
      if ((v_ampl[depth] > 30. && (subdet_ != "HE" || subdet_ != "HB")) ||
          (v_ampl[depth] > 300.)) {  //300 fC cut for QIE-11 HB & HE
        double fbinSOI = tool[soi] - calibrations.pedestal((dataFrame)[soi].capid());
        double fbinPS = 0;

        for (int j = soi + 1; j <= lastbin; j++)
          fbinPS += tool[j] - calibrations.pedestal((dataFrame)[j].capid());

        fbinSOI /= v_ampl[depth];
        fbinPS /= v_ampl[depth];
        strtmp = "HcalDigiTask_SOI_frac_" + subdet_;
        fill1D(strtmp, fbinSOI);
        strtmp = "HcalDigiTask_postSOI_frac_" + subdet_;
        fill1D(strtmp, fbinPS);
      }

      if (hep17_) {
        if (!isHEP17) {
          strtmp = "HcalDigiTask_signal_amplitude_" + subdet_;
          fill1D(strtmp, v_ampl[0]);
          strtmp = "HcalDigiTask_signal_amplitude_depth" + str(depth) + "_" + subdet_;
          fill1D(strtmp, v_ampl[depth]);
        } else {
          strtmp = "HcalDigiTask_signal_amplitude_HEP17";
          fill1D(strtmp, v_ampl[0]);
          strtmp = "HcalDigiTask_signal_amplitude_depth" + str(depth) + "_HEP17";
          fill1D(strtmp, v_ampl[depth]);
        }
      } else {
        strtmp = "HcalDigiTask_signal_amplitude_" + subdet_;
        fill1D(strtmp, v_ampl[0]);
        strtmp = "HcalDigiTask_signal_amplitude_depth" + str(depth) + "_" + subdet_;
        fill1D(strtmp, v_ampl[depth]);
      }
    }
  }  // End of CYCLE OVER CELLS =============================================

  if (isubdet != 0 && noise_ == 0) {  // signal only, once per event
    strtmp = "HcalDigiTask_number_of_amplitudes_above_10fC_" + subdet_;
    fill1D(strtmp, indigis);

    // SimHits once again !!!
    double eps = 1.e-3;
    std::vector<double> v_ehits(maxDepth_[isubdet] + 1, 0);

    if (mc_ == "yes") {
      edm::Handle<edm::PCaloHitContainer> hcalHits;
      iEvent.getByToken(tok_mc_, hcalHits);
      const edm::PCaloHitContainer* simhitResult = hcalHits.product();
      for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin(); simhits != simhitResult->end();
           ++simhits) {
        unsigned int id = simhits->id();
        int sub, depth, ieta, iphi;
        HcalDetId hid;
        if (testNumber_)
          hid = HcalHitRelabeller::relabel(id, hcons_);
        else
          hid = HcalDetId(id);
        sub = hid.subdet();
        depth = hid.depth();
        ieta = hid.ieta();
        iphi = hid.iphi();

        if (depth > maxDepth_[isubdet] && sub == isubdet) {
          edm::LogWarning("HcalDetId") << "HcalDetID(SimHit) presents conflicting information. Depth: " << depth
                                       << ", iphi: " << iphi << ", ieta: " << ieta
                                       << ". Max depth from geometry is: " << maxDepth_[isubdet]
                                       << ". TestNumber = " << testNumber_;
          continue;
        }

        // take cell already found to be max energy in a particular subdet
        if (sub == isubdet && ieta == ieta_Sim && iphi == iphi_Sim) {
          double en = simhits->energy();

          v_ehits[0] += en;
          v_ehits[depth] += en;
        }
      }  // simhit loop

      strtmp = "HcalDigiTask_amplitude_vs_simhits_" + subdet_;
      if (v_ehits[0] > eps)
        fill2D(strtmp, v_ehits[0], v_ampl_c[0]);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        strtmp = "HcalDigiTask_amplitude_vs_simhits_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fill2D(strtmp, v_ehits[depth], v_ampl_c[depth]);
      }

      strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_" + subdet_;
      if (v_ehits[0] > eps)
        fillPf(strtmp, v_ehits[0], v_ampl_c[0]);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fillPf(strtmp, v_ehits[depth], v_ampl_c[depth]);
      }

      strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_" + subdet_;
      if (v_ehits[0] > eps)
        fill1D(strtmp, v_ampl_c[0] / v_ehits[0]);
      for (int depth = 1; depth <= maxDepth_[isubdet]; depth++) {
        strtmp = "HcalDigiTask_amplitude_vs_simhits_profile_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fillPf(strtmp, v_ehits[depth], v_ampl_c[depth]);
        strtmp = "HcalDigiTask_ratio_amplitude_vs_simhits_depth" + str(depth) + "_" + subdet_;
        if (v_ehits[depth] > eps)
          fill1D(strtmp, v_ampl_c[depth] / v_ehits[depth]);
      }

    }  // end of if(mc_ == "yes")

    strtmp = "HcalDigiTask_Ndigis_" + subdet_;
    fill1D(strtmp, double(Ndig));

  }  //  end of if( subdet != 0 && noise_ == 0) { // signal only

}  //HcalDataFrameContainer

void HcalDigiStudy::book1D(edm::Service<TFileService>& fs, std::string name, int n, double min, double max) {
  if (hist1_.find(name) != hist1_.end())
    hist1_[name] = fs->make<TH1D>(name.c_str(), name.c_str(), n, min, max);
}

void HcalDigiStudy::book1D(edm::Service<TFileService>& fs, std::string name, const HistLim& limX) {
  if (hist1_.find(name) == hist1_.end())
    hist1_[name] = fs->make<TH1D>(name.c_str(), name.c_str(), limX.n, limX.min, limX.max);
}

void HcalDigiStudy::book2D(edm::Service<TFileService>& fs, std::string name, const HistLim& limX, const HistLim& limY) {
  if (hist2_.find(name) == hist2_.end())
    hist2_[name] = fs->make<TH2D>(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.n, limY.min, limY.max);
}

void HcalDigiStudy::bookPf(edm::Service<TFileService>& fs, std::string name, const HistLim& limX, const HistLim& limY) {
  if (histP_.find(name) == histP_.end())
    histP_[name] = fs->make<TProfile>(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.min, limY.max);
}

void HcalDigiStudy::bookPf(
    edm::Service<TFileService>& fs, std::string name, const HistLim& limX, const HistLim& limY, const char* option) {
  if (histP_.find(name) == histP_.end())
    histP_[name] =
        fs->make<TProfile>(name.c_str(), name.c_str(), limX.n, limX.min, limX.max, limY.min, limY.max, option);
}

void HcalDigiStudy::fill1D(std::string name, double X, double weight) {
  if (hist1_.find(name) != hist1_.end())
    hist1_[name]->Fill(X, weight);
}

void HcalDigiStudy::fill2D(std::string name, double X, double Y, double weight) {
  if (hist2_.find(name) != hist2_.end())
    hist2_[name]->Fill(X, Y, weight);
}

void HcalDigiStudy::fillPf(std::string name, double X, double Y) {
  if (histP_.find(name) != histP_.end())
    histP_[name]->Fill(X, Y);
}

std::string HcalDigiStudy::str(int x) {
  std::stringstream out;
  out << x;
  return out.str();
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDigiStudy);
