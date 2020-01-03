// system include files
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTestBeamNumbering.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalDetId.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalGeometry.h"

#include "SimDataFormats/CaloHit/interface/PassiveHit.h"

// Root objects
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"

//#define EDM_ML_DEBUG

class HGCalTBAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit HGCalTBAnalyzer(edm::ParameterSet const&);
  ~HGCalTBAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void analyzeSimHits(int type, std::vector<PCaloHit>& hits, double zFront);
  void analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk,
                        edm::Handle<edm::SimVertexContainer> const& SimVtx);
  template <class T1>
  void analyzeDigi(int type, const T1& detId, uint16_t adc);
  void analyzeRecHits(int type, edm::Handle<HGCRecHitCollection>& hits);
  void analyzePassiveHits(edm::Handle<edm::PassiveHitContainer> const& hgcPh, int subdet);
  static bool sortTime(const std::pair<double, double>& i, const std::pair<double, double>& j);

  edm::Service<TFileService> fs_;
  std::unique_ptr<AHCalGeometry> ahcalGeom_;
  const HGCalDDDConstants* hgcons_[2];
  const HGCalGeometry* hgeom_[2];
  bool ifEE_, ifFH_, ifBH_, ifBeam_;
  bool doTree_, doTreeCell_;
  bool doSimHits_, doDigis_, doRecHits_;
  bool doPassive_, doPassiveEE_, doPassiveHE_, doPassiveBH_;
  std::string detectorEE_, detectorFH_;
  std::string detectorBH_, detectorBeam_;
  double zFrontEE_, zFrontFH_, zFrontBH_;
  int sampleIndex_;
  std::vector<int> idBeams_;
  double gev2mip200_, gev2mip300_, stoc_smear_time_200_, stoc_smear_time_300_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hitsEE_, tok_hitsFH_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hitsBH_, tok_hitsBeam_;
  edm::EDGetTokenT<edm::SimTrackContainer> tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;
  edm::EDGetToken tok_digiEE_, tok_digiFH_, tok_digiBH_;
  edm::EDGetToken tok_hitrEE_, tok_hitrFH_, tok_hitrBH_;
  edm::EDGetTokenT<edm::HepMCProduct> tok_hepMC_;
  edm::EDGetTokenT<edm::PassiveHitContainer> tok_hgcPHEE_, tok_hgcPHFH_;
  edm::EDGetTokenT<edm::PassiveHitContainer> tok_hgcPHBH_, tok_hgcPHCMSE_, tok_hgcPHBeam_;

  TTree* tree_;
  TH1D *hSimHitE_[4], *hSimHitT_[4];
  TH1D *hDigiADC_[3], *hDigiLng_[2];
  TH1D *hRecHitE_[3], *hSimHitEn_[4], *hBeam_;
  TH2D *hDigiOcc_[3], *hRecHitOcc_[3];
  TProfile *hSimHitLng_[3], *hSimHitLng1_[3];
  TProfile* hSimHitLng2_[3];
  TProfile *hRecHitLng_[3], *hRecHitLng1_[3];
  TProfile2D *hSimHitLat_[3], *hRecHitLat_[3];
  std::vector<TH1D*> hSimHitLayEn1EE_, hSimHitLayEn2EE_;
  std::vector<TH1D*> hSimHitLayEn1FH_, hSimHitLayEn2FH_;
  std::vector<TH1D*> hSimHitLayEn1BH_, hSimHitLayEn2BH_;
  std::vector<TH1D*> hSimHitLayEnBeam_;
  std::vector<float> simHitLayEn1EE_, simHitLayEn2EE_;
  std::vector<float> simHitLayEn1FH_, simHitLayEn2FH_;
  std::vector<float> simHitLayEn1BH_, simHitLayEn2BH_;
  std::vector<float> simHitLayEnBeam_;
  std::vector<uint32_t> simHitCellIdEE_, simHitCellIdFH_;
  std::vector<uint32_t> simHitCellIdBH_, simHitCellIdBeam_;
  std::vector<float> simHitCellEnEE_, simHitCellEnFH_;
  std::vector<float> simHitCellEnBH_, simHitCellEnBeam_;
  std::vector<int> simHitCellColBH_, simHitCellRowBH_, simHitCellLayerBH_;
  std::vector<float> simHitCellTimeFirstHitEE_, simHitCellTimeFirstHitFH_, simHitCellTimeFirstHitBH_;
  std::vector<float> simHitCellTime15MipEE_, simHitCellTime15MipFH_, simHitCellTime15MipBH_;
  std::vector<float> simHitCellTimeLastHitEE_, simHitCellTimeLastHitFH_, simHitCellTimeLastHitBH_;

  std::vector<float> hgcPassiveEEEnergy_, hgcPassiveFHEnergy_, hgcPassiveBHEnergy_, hgcPassiveCMSEEnergy_;
  std::vector<float> hgcPassiveBeamEnergy_;
  std::vector<std::string> hgcPassiveEEName_, hgcPassiveFHName_, hgcPassiveBHName_, hgcPassiveCMSEName_;
  std::vector<std::string> hgcPassiveBeamName_;
  std::vector<int> hgcPassiveEEID_, hgcPassiveFHID_, hgcPassiveBHID_, hgcPassiveCMSEID_, hgcPassiveBeamID_;

  double xBeam_, yBeam_, zBeam_, pBeam_;
  double thetaBeam_, phiBeam_;
};

HGCalTBAnalyzer::HGCalTBAnalyzer(const edm::ParameterSet& iConfig) {
  usesResource("TFileService");
  ahcalGeom_.reset(new AHCalGeometry(iConfig));

  // now do whatever initialization is needed
  detectorEE_ = iConfig.getParameter<std::string>("detectorEE");
  detectorFH_ = iConfig.getParameter<std::string>("detectorFH");
  detectorBH_ = iConfig.getParameter<std::string>("detectorBH");
  detectorBeam_ = iConfig.getParameter<std::string>("detectorBeam");
  ifEE_ = iConfig.getParameter<bool>("useEE");
  ifFH_ = iConfig.getParameter<bool>("useFH");
  ifBH_ = iConfig.getParameter<bool>("useBH");
  ifBeam_ = iConfig.getParameter<bool>("useBeam");
  zFrontEE_ = iConfig.getParameter<double>("zFrontEE");
  zFrontFH_ = iConfig.getParameter<double>("zFrontFH");
  zFrontBH_ = iConfig.getParameter<double>("zFrontBH");
  idBeams_ = iConfig.getParameter<std::vector<int>>("idBeams");
  doSimHits_ = iConfig.getParameter<bool>("doSimHits");
  doDigis_ = iConfig.getParameter<bool>("doDigis");
  sampleIndex_ = iConfig.getParameter<int>("sampleIndex");
  doRecHits_ = iConfig.getParameter<bool>("doRecHits");
  doTree_ = iConfig.getParameter<bool>("doTree");
  doTreeCell_ = iConfig.getParameter<bool>("doTreeCell");
  doPassive_ = iConfig.getParameter<bool>("doPassive");
  doPassiveEE_ = iConfig.getParameter<bool>("doPassiveEE");
  doPassiveHE_ = iConfig.getParameter<bool>("doPassiveHE");
  doPassiveBH_ = iConfig.getParameter<bool>("doPassiveBH");
  gev2mip200_ = iConfig.getUntrackedParameter<double>("gev2mip200", 57.0e-6);
  gev2mip300_ = iConfig.getUntrackedParameter<double>("gev2mip300", 85.5e-6);
  ///TIME SMEARING
  ///Const = 50ps = 0.05ns
  ///Stochastic is 4ns/Q(fC)
  ////For 300 um, 1MIP = 3.84fC, 200um, it is 3.84*2./3.=2.56fC
  ////This stochastic for 300um = 4*3.84/E_MIP=15.5ns/E_MIP and 200um = 4*2.56/E_MIPs=10.24ns/E_MIP
  stoc_smear_time_200_ = iConfig.getUntrackedParameter<double>("stoc_smear_time_200", 10.24);
  stoc_smear_time_300_ = iConfig.getUntrackedParameter<double>("stoc_smear_time_300", 15.5);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: SimHits = " << doSimHits_ << " Digis = " << doDigis_ << ":"
                             << sampleIndex_ << " RecHits = " << doRecHits_ << " useDets " << ifEE_ << ":" << ifFH_
                             << ":" << ifBH_ << ":" << ifBeam_ << " zFront " << zFrontEE_ << ":" << zFrontFH_ << ":"
                             << zFrontBH_ << " IdBeam " << idBeams_.size() << ":";
  for (unsigned int k = 0; k < idBeams_.size(); ++k)
    edm::LogVerbatim("HGCSim") << " [" << k << "] " << idBeams_[k];
  edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: DoPassive " << doPassive_ << ":" << doPassiveEE_ << ":"
                             << doPassiveHE_ << ":" << doPassiveBH_;
  edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: MIP conversion factors " << gev2mip200_ << ":" << gev2mip300_
                             << " Time smearing " << stoc_smear_time_200_ << ":" << stoc_smear_time_300_;
#endif
  if (idBeams_.empty())
    idBeams_.push_back(1001);

  edm::InputTag tmp0 = iConfig.getParameter<edm::InputTag>("generatorSrc");
  tok_hepMC_ = consumes<edm::HepMCProduct>(tmp0);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: GeneratorSource = " << tmp0;
#endif
  std::string tmp1 = iConfig.getParameter<std::string>("caloHitSrcEE");
  tok_hitsEE_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", tmp1));
  tok_simTk_ = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  tok_simVtx_ = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  edm::InputTag tmp2 = iConfig.getParameter<edm::InputTag>("digiSrcEE");
  tok_digiEE_ = consumes<HGCalDigiCollection>(tmp2);
  edm::InputTag tmp3 = iConfig.getParameter<edm::InputTag>("recHitSrcEE");
  tok_hitrEE_ = consumes<HGCRecHitCollection>(tmp3);
#ifdef EDM_ML_DEBUG
  if (ifEE_) {
    edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: Detector " << detectorEE_ << " with tags " << tmp1 << ", " << tmp2
                               << ", " << tmp3;
  }
#endif
  tmp1 = iConfig.getParameter<std::string>("caloHitSrcFH");
  tok_hitsFH_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", tmp1));
  tmp2 = iConfig.getParameter<edm::InputTag>("digiSrcFH");
  tok_digiFH_ = consumes<HGCalDigiCollection>(tmp2);
  tmp3 = iConfig.getParameter<edm::InputTag>("recHitSrcFH");
  tok_hitrFH_ = consumes<HGCRecHitCollection>(tmp3);
#ifdef EDM_ML_DEBUG
  if (ifFH_) {
    edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: Detector " << detectorFH_ << " with tags " << tmp1 << ", " << tmp2
                               << ", " << tmp3;
  }
#endif
  tmp1 = iConfig.getParameter<std::string>("caloHitSrcBH");
  tok_hitsBH_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", tmp1));
  tmp2 = iConfig.getParameter<edm::InputTag>("digiSrcBH");
  tok_digiBH_ = consumes<HGCalDigiCollection>(tmp2);
  tmp3 = iConfig.getParameter<edm::InputTag>("recHitSrcBH");
  tok_hitrBH_ = consumes<HGCRecHitCollection>(tmp3);

  /// Passive hits
  edm::InputTag tmp = iConfig.getParameter<edm::InputTag>("passiveEE");
  tok_hgcPHEE_ = consumes<edm::PassiveHitContainer>(tmp);

  tmp = iConfig.getParameter<edm::InputTag>("passiveFH");
  tok_hgcPHFH_ = consumes<edm::PassiveHitContainer>(tmp);

  tmp = iConfig.getParameter<edm::InputTag>("passiveBH");
  tok_hgcPHBH_ = consumes<edm::PassiveHitContainer>(tmp);

  tmp = iConfig.getParameter<edm::InputTag>("passiveCMSE");
  tok_hgcPHCMSE_ = consumes<edm::PassiveHitContainer>(tmp);

  tmp = iConfig.getParameter<edm::InputTag>("passiveBeam");
  tok_hgcPHBeam_ = consumes<edm::PassiveHitContainer>(tmp);

#ifdef EDM_ML_DEBUG
  if (ifBH_) {
    edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: Detector " << detectorBH_ << " with tags " << tmp1 << ", " << tmp2
                               << ", " << tmp3;
  }
#endif
  tmp1 = iConfig.getParameter<std::string>("caloHitSrcBeam");
  tok_hitsBeam_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", tmp1));
#ifdef EDM_ML_DEBUG
  if (ifBeam_) {
    edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer:: Detector " << detectorBeam_ << " with tags " << tmp1;
  }
#endif
}

HGCalTBAnalyzer::~HGCalTBAnalyzer() {}

void HGCalTBAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detectorEE", "HGCalEESensitive");
  desc.add<bool>("useEE", true);
  desc.add<double>("zFrontEE", 0.0);
  desc.add<std::string>("caloHitSrcEE", "HGCHitsEE");
  desc.add<edm::InputTag>("digiSrcEE", edm::InputTag("hgcalDigis", "EE"));
  desc.add<edm::InputTag>("recHitSrcEE", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<std::string>("detectorFH", "HGCalHESiliconSensitive");
  desc.add<bool>("useFH", false);
  desc.add<double>("zFrontFH", 0.0);
  desc.add<std::string>("caloHitSrcFH", "HGCHitsHEfront");
  desc.add<edm::InputTag>("digiSrcFH", edm::InputTag("hgcalDigis", "HEfront"));
  desc.add<edm::InputTag>("recHitSrcFH", edm::InputTag("HGCalRecHit", "HGCHEFRecHits"));
  desc.add<std::string>("detectorBH", "AHCal");
  desc.add<bool>("useBH", false);
  desc.add<double>("zFrontBH", 0.0);
  desc.add<std::string>("caloHitSrcBH", "HcalHits");
  desc.add<edm::InputTag>("digiSrcBH", edm::InputTag("hgcalDigis", "HEback"));
  desc.add<edm::InputTag>("recHitSrcBH", edm::InputTag("HGCalRecHit", "HGCHEBRecHits"));
  desc.add<std::string>("detectorBeam", "HcalTB06BeamDetector");
  desc.add<bool>("useBeam", false);
  desc.add<std::string>("caloHitSrcBeam", "HcalTB06BeamHits");
  std::vector<int> ids = {
      1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1011, 1012, 1013, 1014, 2001, 2002, 2003, 2004, 2005};
  desc.add<std::vector<int>>("idBeams", ids);
  desc.add<edm::InputTag>("generatorSrc", edm::InputTag("generatorSmeared"));
  desc.add<edm::InputTag>("passiveEE", edm::InputTag("g4SimHits", "HGCalEEPassiveHits"));
  desc.add<edm::InputTag>("passiveFH", edm::InputTag("g4SimHits", "HGCalHEPassiveHits"));
  desc.add<edm::InputTag>("passiveBH", edm::InputTag("g4SimHits", "HGCalAHPassiveHits"));
  desc.add<edm::InputTag>("passiveCMSE", edm::InputTag("g4SimHits", "CMSEPassiveHits"));
  desc.add<edm::InputTag>("passiveBeam", edm::InputTag("g4SimHits", "HGCalBeamPassiveHits"));

  desc.add<bool>("doSimHits", true);
  desc.add<bool>("doDigis", true);
  desc.add<bool>("doRecHits", true);
  desc.add<int>("sampleIndex", 0);
  desc.add<bool>("doTree", true);
  desc.add<bool>("doTreeCell", true);
  desc.add<bool>("doPassive", false);
  desc.add<bool>("doPassiveEE", false);
  desc.add<bool>("doPassiveHE", false);
  desc.add<bool>("doPassiveBH", false);
  desc.addUntracked<double>("gev2mip200", 57.0e-6);
  desc.addUntracked<double>("gev2mip300", 85.5e-6);
  desc.addUntracked<double>("stoc_smear_time_200", 10.24);
  desc.addUntracked<double>("stoc_smear_time_300", 15.5);
  desc.addUntracked<int>("maxDepth", 12);
  desc.addUntracked<double>("deltaX", 30.0);  // Size of tile along X
  desc.addUntracked<double>("deltaY", 30.0);  // Size of tile along Y
  desc.addUntracked<double>("deltaZ", 81.0);  // Thickness of a single layer
  desc.addUntracked<double>("zFirst", 17.6);  // Position of the center
  descriptions.add("HGCalTBAnalyzer", desc);
}

void HGCalTBAnalyzer::beginJob() {
  char name[40], title[100];
  hBeam_ = fs_->make<TH1D>("BeamP", "Beam Momentum", 1000, 0, 1000.0);
  for (int i = 0; i < 3; ++i) {
    bool book(ifEE_);
    std::string det(detectorEE_);
    if (i == 1) {
      book = ifFH_;
      det = detectorFH_;
    } else if (i == 2) {
      book = ifBH_;
      det = detectorBH_;
    }

    if (doSimHits_ && book) {
      sprintf(name, "SimHitEn%s", det.c_str());
      sprintf(title, "Sim Hit Energy for %s", det.c_str());
      hSimHitE_[i] = fs_->make<TH1D>(name, title, 100000, 0., 0.2);
      sprintf(name, "SimHitEnX%s", det.c_str());
      sprintf(title, "Sim Hit Energy for %s", det.c_str());
      hSimHitEn_[i] = fs_->make<TH1D>(name, title, 100000, 0., 0.2);
      sprintf(name, "SimHitTm%s", det.c_str());
      sprintf(title, "Sim Hit Timing for %s", det.c_str());
      hSimHitT_[i] = fs_->make<TH1D>(name, title, 5000, 0., 500.0);
      sprintf(name, "SimHitLat%s", det.c_str());
      sprintf(title, "Lateral Shower profile (Sim Hit) for %s", det.c_str());
      hSimHitLat_[i] = fs_->make<TProfile2D>(name, title, 100, -100., 100., 100, -100., 100.);
      sprintf(name, "SimHitLng%s", det.c_str());
      sprintf(title, "Longitudinal Shower profile (Sim Hit) for %s", det.c_str());
      hSimHitLng_[i] = fs_->make<TProfile>(name, title, 50, 0., 100.);
      sprintf(name, "SimHitLng1%s", det.c_str());
      sprintf(title, "Longitudinal Shower profile (Layer) for %s", det.c_str());
      hSimHitLng1_[i] = fs_->make<TProfile>(name, title, 200, 0., 100.);
      sprintf(name, "SimHitLng2%s", det.c_str());
      sprintf(title, "Longitudinal Shower profile (Layer) for %s", det.c_str());
      hSimHitLng2_[i] = fs_->make<TProfile>(name, title, 200, 0., 100.);
    }

    if (doDigis_ && book) {
      sprintf(name, "DigiADC%s", det.c_str());
      sprintf(title, "ADC at Digi level for %s", det.c_str());
      hDigiADC_[i] = fs_->make<TH1D>(name, title, 100, 0., 100.0);
      sprintf(name, "DigiOcc%s", det.c_str());
      sprintf(title, "Occupancy (Digi)for %s", det.c_str());
      hDigiOcc_[i] = fs_->make<TH2D>(name, title, 100, -10., 10., 100, -10., 10.);
      sprintf(name, "DigiLng%s", det.c_str());
      sprintf(title, "Longitudinal Shower profile (Digi) for %s", det.c_str());
      hDigiLng_[i] = fs_->make<TH1D>(name, title, 100, 0., 10.);
    }

    if (doRecHits_ && book) {
      sprintf(name, "RecHitEn%s", det.c_str());
      sprintf(title, "Rec Hit Energy for %s", det.c_str());
      hRecHitE_[i] = fs_->make<TH1D>(name, title, 1000, 0., 10.0);
      sprintf(name, "RecHitOcc%s", det.c_str());
      sprintf(title, "Occupancy (Rec Hit)for %s", det.c_str());
      hRecHitOcc_[i] = fs_->make<TH2D>(name, title, 100, -10., 10., 100, -10., 10.);
      sprintf(name, "RecHitLat%s", det.c_str());
      sprintf(title, "Lateral Shower profile (Rec Hit) for %s", det.c_str());
      hRecHitLat_[i] = fs_->make<TProfile2D>(name, title, 100, -10., 10., 100, -10., 10.);
      sprintf(name, "RecHitLng%s", det.c_str());
      sprintf(title, "Longitudinal Shower profile (Rec Hit) for %s", det.c_str());
      hRecHitLng_[i] = fs_->make<TProfile>(name, title, 100, 0., 10.);
      sprintf(name, "RecHitLng1%s", det.c_str());
      sprintf(title, "Longitudinal Shower profile vs Layer for %s", det.c_str());
      hRecHitLng1_[i] = fs_->make<TProfile>(name, title, 120, 0., 60.);
    }
  }
  if (ifBeam_ && doSimHits_) {
    sprintf(name, "SimHitEn%s", detectorBeam_.c_str());
    sprintf(title, "Sim Hit Energy for %s", detectorBeam_.c_str());
    hSimHitE_[3] = fs_->make<TH1D>(name, title, 100000, 0., 0.2);
    sprintf(name, "SimHitEnX%s", detectorBeam_.c_str());
    sprintf(title, "Sim Hit Energy for %s", detectorBeam_.c_str());
    hSimHitEn_[3] = fs_->make<TH1D>(name, title, 100000, 0., 0.2);
    sprintf(name, "SimHitTm%s", detectorBeam_.c_str());
    sprintf(title, "Sim Hit Timing for %s", detectorBeam_.c_str());
    hSimHitT_[3] = fs_->make<TH1D>(name, title, 5000, 0., 500.0);
  }
  if (doSimHits_ && doTree_) {
    tree_ = fs_->make<TTree>("HGCTB", "SimHitEnergy");
    tree_->Branch("simHitLayEn1EE", &simHitLayEn1EE_);
    tree_->Branch("simHitLayEn2EE", &simHitLayEn2EE_);
    tree_->Branch("simHitLayEn1FH", &simHitLayEn1FH_);
    tree_->Branch("simHitLayEn2FH", &simHitLayEn2FH_);
    tree_->Branch("simHitLayEn1BH", &simHitLayEn1BH_);
    tree_->Branch("simHitLayEn2BH", &simHitLayEn2BH_);
    tree_->Branch("xBeam", &xBeam_, "xBeam/D");
    tree_->Branch("yBeam", &yBeam_, "yBeam/D");
    tree_->Branch("zBeam", &zBeam_, "zBeam/D");
    tree_->Branch("pBeam", &pBeam_, "pBeam/D");
    tree_->Branch("thetaBeam", &thetaBeam_, "thetaBeam/D");
    tree_->Branch("phiBeam", &phiBeam_, "phiBeam/D");
    if (doTreeCell_) {
      tree_->Branch("simHitCellIdEE", &simHitCellIdEE_);
      tree_->Branch("simHitCellEnEE", &simHitCellEnEE_);
      tree_->Branch("simHitCellIdFH", &simHitCellIdFH_);
      tree_->Branch("simHitCellEnFH", &simHitCellEnFH_);
      tree_->Branch("simHitCellIdBH", &simHitCellIdBH_);
      tree_->Branch("simHitCellEnBH", &simHitCellEnBH_);
      tree_->Branch("simHitCellIdBeam", &simHitCellIdBeam_);
      tree_->Branch("simHitCellEnBeam", &simHitCellEnBeam_);

      tree_->Branch("simHitCellColBH", &simHitCellColBH_);
      tree_->Branch("simHitCellRowBH", &simHitCellRowBH_);
      tree_->Branch("simHitCellLayerBH", &simHitCellLayerBH_);
      tree_->Branch("simHitCellTimeFirstHitEE", &simHitCellTimeFirstHitEE_);
      tree_->Branch("simHitCellTimeFirstHitFH", &simHitCellTimeFirstHitFH_);
      //tree_->Branch("simHitCellTimeFirstHitBH", &simHitCellTimeFirstHitBH_);
      tree_->Branch("simHitCellTime15MipEE", &simHitCellTime15MipEE_);
      tree_->Branch("simHitCellTime15MipFH", &simHitCellTime15MipFH_);
      //tree_->Branch("simHitCellTime15MipBH", &simHitCellTime15MipBH_);
      tree_->Branch("simHitCellTimeLastHitEE", &simHitCellTimeLastHitEE_);
      tree_->Branch("simHitCellTimeLastHitFH", &simHitCellTimeLastHitFH_);
      //tree_->Branch("simHitCellTimeLastHitBH", &simHitCellTimeLastHitBH_);
    }
  }

  if (doPassive_ && doTree_) {
    if (doPassiveEE_) {
      tree_->Branch("hgcPassiveEEEnergy", &hgcPassiveEEEnergy_);
      tree_->Branch("hgcPassiveEEName", &hgcPassiveEEName_);
      tree_->Branch("hgcPassiveEEID", &hgcPassiveEEID_);
    }
    if (doPassiveHE_) {
      tree_->Branch("hgcPassiveFHEnergy", &hgcPassiveFHEnergy_);
      tree_->Branch("hgcPassiveFHName", &hgcPassiveFHName_);
      tree_->Branch("hgcPassiveFHID", &hgcPassiveFHID_);
    }
    if (doPassiveBH_) {
      tree_->Branch("hgcPassiveBHEnergy", &hgcPassiveBHEnergy_);
      tree_->Branch("hgcPassiveBHName", &hgcPassiveBHName_);
      tree_->Branch("hgcPassiveBHID", &hgcPassiveBHID_);
    }
    tree_->Branch("hgcPassiveCMSEEnergy", &hgcPassiveCMSEEnergy_);
    tree_->Branch("hgcPassiveCMSEName", &hgcPassiveCMSEName_);
    tree_->Branch("hgcPassiveCMSEID", &hgcPassiveCMSEID_);
    tree_->Branch("hgcPassiveBeamEnergy", &hgcPassiveBeamEnergy_);
    tree_->Branch("hgcPassiveBeamName", &hgcPassiveBeamName_);
    tree_->Branch("hgcPassiveBeamID", &hgcPassiveBeamID_);
  }
}

void HGCalTBAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  char name[40], title[100];
  if (ifEE_) {
    edm::ESHandle<HGCalDDDConstants> pHGDC;
    iSetup.get<IdealGeometryRecord>().get(detectorEE_, pHGDC);
    hgcons_[0] = &(*pHGDC);
    if (doDigis_ || doRecHits_) {
      edm::ESHandle<HGCalGeometry> geom;
      iSetup.get<IdealGeometryRecord>().get(detectorEE_, geom);
      hgeom_[0] = geom.product();
    } else {
      hgeom_[0] = nullptr;
    }
    for (unsigned int l = 0; l < hgcons_[0]->layers(false); ++l) {
      sprintf(name, "SimHitEnA%d%s", l, detectorEE_.c_str());
      sprintf(title, "Sim Hit Energy in SIM layer %d for %s", l + 1, detectorEE_.c_str());
      hSimHitLayEn1EE_.push_back(fs_->make<TH1D>(name, title, 100000, 0., 0.2));
      if (l % 3 == 0) {
        sprintf(name, "SimHitEnB%d%s", (l / 3 + 1), detectorEE_.c_str());
        sprintf(title, "Sim Hit Energy in layer %d for %s", (l / 3 + 1), detectorEE_.c_str());
        hSimHitLayEn2EE_.push_back(fs_->make<TH1D>(name, title, 100000, 0., 0.2));
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer::" << detectorEE_ << " defined with " << hgcons_[0]->layers(false)
                               << " layers";
#endif
  } else {
    hgcons_[0] = nullptr;
    hgeom_[0] = nullptr;
  }

  if (ifFH_) {
    edm::ESHandle<HGCalDDDConstants> pHGDC;
    iSetup.get<IdealGeometryRecord>().get(detectorFH_, pHGDC);
    hgcons_[1] = &(*pHGDC);
    if (doDigis_ || doRecHits_) {
      edm::ESHandle<HGCalGeometry> geom;
      iSetup.get<IdealGeometryRecord>().get(detectorFH_, geom);
      hgeom_[1] = geom.product();
    } else {
      hgeom_[1] = nullptr;
    }
    for (unsigned int l = 0; l < hgcons_[1]->layers(false); ++l) {
      sprintf(name, "SimHitEnA%d%s", l, detectorFH_.c_str());
      sprintf(title, "Sim Hit Energy in layer %d for %s", l + 1, detectorFH_.c_str());
      hSimHitLayEn1FH_.push_back(fs_->make<TH1D>(name, title, 100000, 0., 0.2));
      if (l % 3 == 0) {
        sprintf(name, "SimHitEnB%d%s", (l / 3 + 1), detectorFH_.c_str());
        sprintf(title, "Sim Hit Energy in layer %d for %s", (l / 3 + 1), detectorFH_.c_str());
        hSimHitLayEn2FH_.push_back(fs_->make<TH1D>(name, title, 100000, 0., 0.2));
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer::" << detectorFH_ << " defined with " << hgcons_[1]->layers(false)
                               << " layers";
#endif
  } else {
    hgcons_[1] = nullptr;
    hgeom_[1] = nullptr;
  }

  if (ifBH_) {
    for (int l = 0; l < ahcalGeom_->maxDepth(); ++l) {
      sprintf(name, "SimHitEnA%d%s", l, detectorBH_.c_str());
      sprintf(title, "Sim Hit Energy in layer %d for %s", l + 1, detectorBH_.c_str());
      hSimHitLayEn1BH_.push_back(fs_->make<TH1D>(name, title, 100000, 0., 0.2));
      sprintf(name, "SimHitEnB%d%s", l, detectorBH_.c_str());
      sprintf(title, "Sim Hit Energy in layer %d for %s", l + 1, detectorBH_.c_str());
      hSimHitLayEn2BH_.push_back(fs_->make<TH1D>(name, title, 100000, 0., 0.2));
    }
  }

  if (ifBeam_) {
    for (unsigned int l = 0; l < idBeams_.size(); ++l) {
      sprintf(name, "SimHitEna%d%s", l, detectorBeam_.c_str());
      sprintf(title, "Sim Hit Energy in type %d for %s", idBeams_[l], detectorBeam_.c_str());
      hSimHitLayEnBeam_.push_back(fs_->make<TH1D>(name, title, 100000, 0., 0.2));
    }
  }
}

void HGCalTBAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Generator input
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_hepMC_, evtMC);
  if (!evtMC.isValid()) {
    edm::LogWarning("HGCal") << "no HepMCProduct found";
  } else {
    const HepMC::GenEvent* myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
         ++p, ++k) {
      if (k == 0)
        hBeam_->Fill((*p)->momentum().rho());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "Particle[" << k << "] with p " << (*p)->momentum().rho() << " theta "
                                 << (*p)->momentum().theta() << " phi " << (*p)->momentum().phi();
#endif
    }
  }

  // Now the Simhits
  if (doSimHits_) {
    edm::Handle<edm::SimTrackContainer> SimTk;
    iEvent.getByToken(tok_simTk_, SimTk);
    edm::Handle<edm::SimVertexContainer> SimVtx;
    iEvent.getByToken(tok_simVtx_, SimVtx);
    analyzeSimTracks(SimTk, SimVtx);

    simHitLayEn1EE_.clear();
    simHitLayEn2EE_.clear();
    simHitLayEn1FH_.clear();
    simHitLayEn2FH_.clear();
    simHitLayEn1BH_.clear();
    simHitLayEn2BH_.clear();
    simHitLayEnBeam_.clear();
    simHitCellIdEE_.clear();
    simHitCellEnEE_.clear();
    simHitCellIdFH_.clear();
    simHitCellEnFH_.clear();
    simHitCellIdBH_.clear();
    simHitCellEnBH_.clear();
    simHitCellIdBeam_.clear();
    simHitCellEnBeam_.clear();
    simHitCellColBH_.clear();
    simHitCellRowBH_.clear();
    simHitCellLayerBH_.clear();
    simHitCellTimeFirstHitEE_.clear();
    simHitCellTime15MipEE_.clear();
    simHitCellTimeLastHitEE_.clear();
    simHitCellTimeFirstHitFH_.clear();
    simHitCellTime15MipFH_.clear();
    simHitCellTimeLastHitFH_.clear();
    /*
    simHitCellTimeFirstHitBH_.clear();
    simHitCellTime15MipBH_.clear(); 
    simHitCellTimeLastHitBH_.clear(); 
    */
    edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
    std::vector<PCaloHit> caloHits;
    if (ifEE_) {
      simHitLayEn1EE_ = std::vector<float>(hgcons_[0]->layers(false), 0);
      simHitLayEn2EE_ = std::vector<float>(hgcons_[0]->layers(true), 0);
      iEvent.getByToken(tok_hitsEE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PcalohitContainer for " << detectorEE_ << " has " << theCaloHitContainers->size()
                                   << " hits";
#endif
        caloHits.clear();
        caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), theCaloHitContainers->end());
        analyzeSimHits(0, caloHits, zFrontEE_);
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PCaloHitContainer does not exist for " << detectorEE_ << " !!!";
#endif
      }
    }
    if (ifFH_) {
      simHitLayEn1FH_ = std::vector<float>(hgcons_[1]->layers(false), 0);
      simHitLayEn2FH_ = std::vector<float>(hgcons_[1]->layers(true), 0);
      iEvent.getByToken(tok_hitsFH_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PcalohitContainer for " << detectorFH_ << " has " << theCaloHitContainers->size()
                                   << " hits";
#endif
        caloHits.clear();
        caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), theCaloHitContainers->end());
        analyzeSimHits(1, caloHits, zFrontFH_);
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PCaloHitContainer does not exist for " << detectorFH_ << " !!!";
#endif
      }
    }
    if (ifBH_) {
      simHitLayEn1BH_ = std::vector<float>(ahcalGeom_->maxDepth(), 0);
      simHitLayEn2BH_ = std::vector<float>(ahcalGeom_->maxDepth(), 0);
      iEvent.getByToken(tok_hitsBH_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PcalohitContainer for " << detectorBH_ << " has " << theCaloHitContainers->size()
                                   << " hits";
#endif
        caloHits.clear();
        caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), theCaloHitContainers->end());
        analyzeSimHits(2, caloHits, zFrontBH_);
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PCaloHitContainer does not exist for " << detectorBH_ << " !!!";
#endif
      }
    }
    if (ifBeam_) {
      simHitLayEnBeam_ = std::vector<float>(idBeams_.size(), 0);
      iEvent.getByToken(tok_hitsBeam_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PcalohitContainer for " << detectorBeam_ << " has "
                                   << theCaloHitContainers->size() << " hits";
#endif
        caloHits.clear();
        caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), theCaloHitContainers->end());
        analyzeSimHits(3, caloHits, 0.0);
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "PCaloHitContainer does not exist for " << detectorBeam_ << " !!!";
#endif
      }
    }
    // if (doTree_) tree_->Fill();
  }  // if (doSimHits_)

  ////Store the info about the Passive hits
  if (doPassive_) {
    /// EE
    hgcPassiveEEEnergy_.clear();
    hgcPassiveEEName_.clear();
    hgcPassiveEEID_.clear();
    if (doPassiveEE_) {
      edm::Handle<edm::PassiveHitContainer> hgcPHEE;
      iEvent.getByToken(tok_hgcPHEE_, hgcPHEE);
      analyzePassiveHits(hgcPHEE, 1);
    }
    /// FH
    hgcPassiveFHEnergy_.clear();
    hgcPassiveFHName_.clear();
    hgcPassiveFHID_.clear();
    if (doPassiveHE_) {
      edm::Handle<edm::PassiveHitContainer> hgcPHFH;
      iEvent.getByToken(tok_hgcPHFH_, hgcPHFH);
      analyzePassiveHits(hgcPHFH, 2);
    }
    /// BH
    hgcPassiveBHEnergy_.clear();
    hgcPassiveBHName_.clear();
    hgcPassiveBHID_.clear();
    if (doPassiveBH_) {
      edm::Handle<edm::PassiveHitContainer> hgcPHBH;
      iEvent.getByToken(tok_hgcPHBH_, hgcPHBH);
      analyzePassiveHits(hgcPHBH, 3);
    }
    /// CMSE
    hgcPassiveCMSEEnergy_.clear();
    hgcPassiveCMSEName_.clear();
    hgcPassiveCMSEID_.clear();
    edm::Handle<edm::PassiveHitContainer> hgcPHCMSE;
    iEvent.getByToken(tok_hgcPHCMSE_, hgcPHCMSE);
    analyzePassiveHits(hgcPHCMSE, 4);
    ///Beam
    hgcPassiveBeamName_.clear();
    hgcPassiveBeamEnergy_.clear();
    hgcPassiveBeamID_.clear();
    edm::Handle<edm::PassiveHitContainer> hgcPHBeam;
    iEvent.getByToken(tok_hgcPHBeam_, hgcPHBeam);
    analyzePassiveHits(hgcPHBeam, 5);
  }

  if ((doSimHits_ || doPassive_) && (doTree_))
    tree_->Fill();

  // Now the Digis
  if (doDigis_) {
    if (ifEE_) {
      edm::Handle<HGCalDigiCollection> theDigiContainers;
      iEvent.getByToken(tok_digiEE_, theDigiContainers);
      if (theDigiContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCDigiCintainer for " << detectorEE_ << " with " << theDigiContainers->size()
                                   << " element(s)";
#endif
        for (const auto& it : *theDigiContainers) {
          HGCalDetId detId = (it.id());
          const HGCSample& hgcSample = it.sample(sampleIndex_);
          uint16_t adc = hgcSample.data();
          analyzeDigi(0, detId, adc);
        }
      }
    }
    if (ifFH_) {
      edm::Handle<HGCalDigiCollection> theDigiContainers;
      iEvent.getByToken(tok_digiFH_, theDigiContainers);
      if (theDigiContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCDigiContainer for " << detectorFH_ << " with " << theDigiContainers->size()
                                   << " element(s)";
#endif
        for (const auto& it : *theDigiContainers) {
          HGCalDetId detId = (it.id());
          const HGCSample& hgcSample = it.sample(sampleIndex_);
          uint16_t adc = hgcSample.data();
          analyzeDigi(1, detId, adc);
        }
      }
    }
  }

  // The Rechits
  if (doRecHits_) {
    edm::Handle<HGCRecHitCollection> theCaloHitContainers;
    if (ifEE_) {
      iEvent.getByToken(tok_hitrEE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCRecHitCollection for " << detectorEE_ << " has "
                                   << theCaloHitContainers->size() << " hits";
#endif
        analyzeRecHits(0, theCaloHitContainers);
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCRecHitCollection does not exist for " << detectorEE_ << " !!!";
#endif
      }
    }
    if (ifFH_) {
      iEvent.getByToken(tok_hitrFH_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCRecHitCollection for " << detectorFH_ << " has "
                                   << theCaloHitContainers->size() << " hits";
#endif
        analyzeRecHits(1, theCaloHitContainers);
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCSim") << "HGCRecHitCollection does not exist for " << detectorFH_ << " !!!";
#endif
      }  // else
    }    // if (ifFH_)
  }      // if (doRecHits_)

}  // void HGCalTBAnalyzer::analyze

void HGCalTBAnalyzer::analyzeSimHits(int type, std::vector<PCaloHit>& hits, double zFront) {
  std::map<uint32_t, double> map_hits, map_hitn;
  std::map<uint32_t, double> map_hittime_firsthit, map_hittime_lasthit, map_hittime_15Mip;
  std::map<int, double> map_hitDepth;
  std::map<int, std::pair<uint32_t, double>> map_hitLayer, map_hitCell;
  double entot(0);
  std::map<uint32_t, double> nhits;
  std::map<uint32_t, int> ID, Depth;
  std::map<uint32_t, double> GeV2Mip;
  std::map<uint32_t, double> StochTermTime;
  std::vector<int> nSimLayers;
  ///storing time of each pcalohit in each cell ///SJ
  std::map<uint32_t, std::vector<std::pair<double, double>>> map_hitTimeEn;
  //bool debug = true;
  bool debug = false;
  for (unsigned int i = 0; i < hits.size(); i++) {
    double energy = hits[i].energy();
    double time = hits[i].time();
    uint32_t id = hits[i].id();
    entot += energy;
    int subdet, zside, layer, sector, subsector(0), cell, depth(0), idx(0);
    if (type == 2) {
      subdet = HcalDetId(id).subdet();
      if (subdet != HcalOther)
        continue;
      AHCalDetId hid(id);
      layer = depth = hid.depth();
      zside = hid.zside();
      sector = hid.irow();
      cell = hid.icol();
      idx = ((hid.irowAbs() * 100) + (hid.icolAbs()));
      if (debug)
        edm::LogVerbatim("HGCSim") << "depth, sector, cell " << depth << ":" << sector << ":" << cell;
    } else if (type == 3) {
      HcalTestBeamNumbering::unpackIndex(id, subdet, layer, sector, cell);
      depth = layer;
      zside = 1;
      idx = subdet * 1000 + layer;
      layer = idx;
    } else {
      HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector, subsector, cell);
      depth = hgcons_[type]->simToReco(cell, layer, sector, true).second;
      idx = sector * 1000 + cell;
#ifdef EDM_ML_DEBUG
      std::pair<float, float> xy = hgcons_[type]->locateCell(cell, layer, sector, false);
      edm::LogVerbatim("HGCSim") << "detId " << std::hex << id << std::dec << " Layer:Wafer:Cell " << layer << ":"
                                 << sector << ":" << cell << " Position " << xy.first << ":" << xy.second << ":"
                                 << hgcons_[type]->waferZ(layer, false);
#endif
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "SimHit:Hit[" << i << "] Id " << subdet << ":" << zside << ":" << layer << ":"
                               << sector << ":" << subsector << ":" << cell << ":" << depth << " Energy " << energy
                               << " Time " << time;
#endif
    if (map_hits.count(id) != 0) {
      map_hits[id] += energy;
    } else {
      map_hits[id] = energy;
    }
    if (map_hitLayer.count(layer) != 0) {
      double ee = energy + map_hitLayer[layer].second;
      map_hitLayer[layer] = std::make_pair(id, ee);
    } else {
      map_hitLayer[layer] = std::make_pair(id, energy);
    }
    if (depth >= 0) {
      if (map_hitCell.count(idx) != 0) {
        double ee = energy + map_hitCell[idx].second;
        map_hitCell[idx] = std::make_pair(id, ee);
      } else {
        map_hitCell[idx] = std::make_pair(id, energy);
      }
      if (debug) {
        if (type == 1)
          edm::LogVerbatim("HGCSim") << "EE, depth is and map_hitDepth[depth] " << depth << " " << map_hitDepth[depth];
        if (type == 2)
          edm::LogVerbatim("HGCSim") << "BH, depth is and map_hitDepth[depth] " << depth << " " << map_hitDepth[depth];
      }
      if (map_hitDepth.count(depth) != 0) {
        map_hitDepth[depth] += energy;
      } else {
        map_hitDepth[depth] = energy;
      }
      uint32_t idn =
          (type >= 2) ? id : HGCalTestNumbering::packHexagonIndex(subdet, zside, depth, sector, subsector, cell);
      map_hitTimeEn[idn].push_back(std::make_pair(time, energy));
      GeV2Mip[idn] = gev2mip300_;
      StochTermTime[idn] = stoc_smear_time_300_;
      ID[idn] = id;
      Depth[idn] = depth;
      if (map_hitn.count(idn) != 0) {
        map_hitn[idn] += energy;
        ++nhits[idn];
      } else {
        map_hitn[idn] = energy;
        nhits[idn] = 1;
      }
    }
    hSimHitT_[type]->Fill(time, energy);
  }

  if (type < 2) {  //store only for EE and FH
    ///now sort the vector of each cell hits
    for (const auto& itr : map_hitTimeEn) {
      uint32_t id = itr.first;
      int wafer = -99;
      ///id: reco and ID[id]: sim ID
      wafer = HGCalDetId(ID[id]).wafer();
      double layer = HGCalDetId(id).layer();
      double thickness = hgcons_[type]->cellThickness(layer, wafer, 0);
      if (debug) {
        edm::LogVerbatim("HGCSim") << "wafer is : depth (reco) " << wafer << " " << Depth[id]
                                   << "\ntype : layer : wafer thickness " << type << " " << layer << " " << thickness
                                   << "\nID(sim) and id(reco) " << std::hex << ID[id] << " " << id << std::dec
                                   << std::endl;
      }
      if (thickness == 300) {
        GeV2Mip[id] = gev2mip300_;
        StochTermTime[id] = stoc_smear_time_300_;
      } else if (thickness == 200) {
        GeV2Mip[id] = gev2mip200_;
        StochTermTime[id] = stoc_smear_time_200_;
      }
      //first sort
      std::sort(map_hitTimeEn[id].begin(), map_hitTimeEn[id].end(), sortTime);
      ///once it is sorted now start adding the energy
      double threshold = 15.;
      double totE = 0.;
      double totEbeforeThreshold = 0.;
      double timebeforeThreshold = 0.;
      double timeAtThresohld = 0.;
      for (unsigned int ihit = 0; ihit < map_hitTimeEn[id].size(); ihit++) {
        double energy = (map_hitTimeEn[id].at(ihit)).second / GeV2Mip[id];
        totE += energy;
        double time = (map_hitTimeEn[id].at(ihit)).first;
        if (debug)
          edm::LogVerbatim("HGCSim") << "Tot E till now : time of that E : GeV2Mip[id] is " << totE << " " << time
                                     << " " << GeV2Mip[id];
        if (totE < threshold) {
          totEbeforeThreshold = totE;
          timebeforeThreshold = time;
        } else {
          timeAtThresohld =
              (threshold - totEbeforeThreshold) * (time - timebeforeThreshold) / (totE - totEbeforeThreshold) +
              timebeforeThreshold;
          map_hittime_15Mip[id] = timeAtThresohld;
          if (debug)
            edm::LogVerbatim("HGCSim") << "ihit : energyBefore : timeBefore : energyTot : timeTot : timeAt15MIP  "
                                       << ihit << " " << totEbeforeThreshold << " " << timebeforeThreshold << " "
                                       << totE << " " << time << " " << map_hittime_15Mip[id];
          break;
        }
      }
      if (!map_hitTimeEn[id].empty()) {
        map_hittime_firsthit[id] = (map_hitTimeEn[id].at(0)).first;
        map_hittime_lasthit[id] = (map_hitTimeEn[id].at(map_hitTimeEn[id].size() - 1)).first;
        if (map_hittime_15Mip[id] < map_hittime_firsthit[id])
          map_hittime_15Mip[id] = map_hittime_firsthit[id];
        /*
        ///Put the smeared values here after correcting for the vertex time
        double firsthit_sm = ran3.Gaus(map_hittime_firsthit_vtxCorr[id], StochTermTime[id]);
        double lasthit_sm = ran3.Gaus(map_hittime_lasthit_vtxCorr[id], StochTermTime[id]);
        double threshmiphit_sm = ran3.Gaus(map_hittime_15Mip_vtxCorr[id], StochTermTime[id]);
        */
      }

      if (totE < threshold)
        map_hittime_15Mip[id] = -99;
      if (debug)
        edm::LogVerbatim("HGCSim") << "id : first hit time :  last hit time " << id << " " << map_hittime_firsthit[id]
                                   << " " << map_hittime_lasthit[id] << "\nFinally for this cell, time is "
                                   << map_hittime_15Mip[id];
    }
  }

  hSimHitEn_[type]->Fill(entot);
  for (const auto& itr : map_hits) {
    hSimHitE_[type]->Fill(itr.second);
  }

  if (debug)
    edm::LogVerbatim("HGCSim") << "Now looping over map_hitLayer";
  for (const auto& itr : map_hitLayer) {
    int layer = (type == 2) ? itr.first : (itr.first - 1);
    double energy = (itr.second).second;
    double zp(0);
    if (type < 2)
      zp = hgcons_[type]->waferZ(layer + 1, false);
    else if (type == 2)
      zp = ahcalGeom_->getZ(AHCalDetId((itr.second).first));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "SimHit:Layer " << layer + 1 << " Z " << zp << ":" << zp - zFront << " E " << energy;
#endif
    if (type < 3) {
      hSimHitLng_[type]->Fill(zp - zFront, energy);
      hSimHitLng2_[type]->Fill(layer + 1, energy);
    }
    if (type == 0) {
      if (layer < static_cast<int>(hSimHitLayEn1EE_.size())) {
        simHitLayEn1EE_[layer] = energy;
        hSimHitLayEn1EE_[layer]->Fill(energy);
      }
    } else if (type == 1) {
      if (layer < static_cast<int>(hSimHitLayEn1FH_.size())) {
        simHitLayEn1FH_[layer] = energy;
        hSimHitLayEn1FH_[layer]->Fill(energy);
      }
    } else if (type == 2) {
      if (debug)
        edm::LogVerbatim("HGCSim") << "layer < hSimHitLayEn1BH_.size()";
      if (layer < static_cast<int>(hSimHitLayEn1BH_.size())) {
        simHitLayEn1BH_[layer] = energy;
        hSimHitLayEn1BH_[layer]->Fill(energy);
      }
    } else {
      for (unsigned int k = 0; k < idBeams_.size(); ++k) {
        if (layer + 1 == idBeams_[k]) {
          simHitLayEnBeam_[k] = energy;
          hSimHitLayEnBeam_[k]->Fill(energy);
          break;
        }
      }
    }
  }
  for (const auto& itr : map_hitDepth) {
    int layer = (type == 2) ? itr.first : (itr.first - 1);
    double energy = itr.second;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "SimHit:Layer " << layer + 1 << " " << energy;
#endif
    hSimHitLng1_[type]->Fill(layer + 1, energy);
    if (type == 0) {
      if (layer < static_cast<int>(hSimHitLayEn2EE_.size())) {
        simHitLayEn2EE_[layer] = energy;
        hSimHitLayEn2EE_[layer]->Fill(energy);
      }
    } else if (type == 1) {
      if (layer < static_cast<int>(hSimHitLayEn2FH_.size())) {
        simHitLayEn2FH_[layer] = energy;
        hSimHitLayEn2FH_[layer]->Fill(energy);
      }
    } else if (type == 2) {
      if (debug)
        edm::LogVerbatim("HGCSim") << "Inside map_hitDepth, layer no. " << layer;
      if (layer < static_cast<int>(hSimHitLayEn2BH_.size())) {
        simHitLayEn2BH_[layer] = energy;
        hSimHitLayEn2BH_[layer]->Fill(energy);
      }
    }
  }

  if (debug)
    edm::LogVerbatim("HGCSim") << "Now looping over map_hitCell";
  if (type < 3) {
    for (const auto& itr : map_hitCell) {
      uint32_t id = ((itr.second).first);
      double energy = ((itr.second).second);
      std::pair<float, float> xy(0, 0);
      double xx(0);
      if (type == 2) {
        xy = ahcalGeom_->getXY(AHCalDetId(id));
        xx = xy.first;
      } else {
        int subdet, zside, layer, sector, subsector, cell;
        HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector, subsector, cell);
        xy = hgcons_[type]->locateCell(cell, layer, sector, false);
        double zp = hgcons_[type]->waferZ(layer, false);
        xx = (zp < 0) ? -xy.first : xy.first;
      }
      hSimHitLat_[type]->Fill(xx, xy.second, energy);
    }
  }

  for (const auto& itr : map_hitn) {
    uint32_t id = itr.first;
    double energy = itr.second;
    if (type == 0) {
      double time_firsthit = map_hittime_firsthit[id];
      double time15Mip = map_hittime_15Mip[id];
      double time_lasthit = map_hittime_lasthit[id];
      simHitCellIdEE_.push_back(id);
      simHitCellEnEE_.push_back(energy);
      simHitCellTimeFirstHitEE_.push_back(time_firsthit);
      simHitCellTime15MipEE_.push_back(time15Mip);
      simHitCellTimeLastHitEE_.push_back(time_lasthit);
      if (debug && (energy / GeV2Mip[id] < 15) && (map_hittime_15Mip[id] > 0))
        edm::LogVerbatim("HGCSim") << "FOUND!!!!rechit energy : Finally for this cell, time is " << energy / GeV2Mip[id]
                                   << " " << map_hittime_15Mip[id];
    } else if (type == 1) {
      double time_firsthit = map_hittime_firsthit[id];
      double time15Mip = map_hittime_15Mip[id];
      double time_lasthit = map_hittime_lasthit[id];
      simHitCellIdFH_.push_back(id);
      simHitCellEnFH_.push_back(energy);
      simHitCellTimeFirstHitFH_.push_back(time_firsthit);
      simHitCellTime15MipFH_.push_back(time15Mip);
      simHitCellTimeLastHitFH_.push_back(time_lasthit);
    } else if (type == 2) {
      simHitCellIdBH_.push_back(id);
      simHitCellEnBH_.push_back(energy);
      AHCalDetId hid(id);
      int row = hid.irow();
      int col = hid.icol();
      int layer = hid.depth();
      simHitCellColBH_.push_back(col);
      simHitCellRowBH_.push_back(row);
      simHitCellLayerBH_.push_back(layer);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCSim") << "ID: " << std::hex << id << std::dec << " Layer: " << layer << " col: " << col
                                 << " row: " << row;
#endif
    } else if (type == 3) {
      simHitCellIdBeam_.push_back(id);
      simHitCellEnBeam_.push_back(energy);
    }
  }
}

void HGCalTBAnalyzer::analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk,
                                       edm::Handle<edm::SimVertexContainer> const& SimVtx) {
  xBeam_ = yBeam_ = zBeam_ = pBeam_ = -1000000;
  thetaBeam_ = phiBeam_ = -100000;
  int vertIndex(-1);
  for (const auto& simTrkItr : *SimTk) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Track " << simTrkItr.trackId() << " Vertex " << simTrkItr.vertIndex() << " Type "
                               << simTrkItr.type() << " Charge " << simTrkItr.charge() << " momentum "
                               << simTrkItr.momentum() << " " << simTrkItr.momentum().P();
#endif
    if (vertIndex == -1) {
      vertIndex = simTrkItr.vertIndex();
      pBeam_ = simTrkItr.momentum().P();
      thetaBeam_ = simTrkItr.momentum().theta();
      phiBeam_ = simTrkItr.momentum().phi();
      if (phiBeam_ < 0)
        phiBeam_ += (2 * M_PI);
    }
  }
  if (vertIndex != -1 && vertIndex < (int)SimVtx->size()) {
    edm::SimVertexContainer::const_iterator simVtxItr = SimVtx->begin();
    for (int iv = 0; iv < vertIndex; iv++)
      simVtxItr++;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Vertex " << vertIndex << " position " << simVtxItr->position();
#endif
    xBeam_ = simVtxItr->position().X();
    yBeam_ = simVtxItr->position().Y();
    zBeam_ = simVtxItr->position().Z();
  }
}

template <class T1>
void HGCalTBAnalyzer::analyzeDigi(int type, const T1& detId, uint16_t adc) {
  DetId id1 = DetId(detId.rawId());
  GlobalPoint global = hgeom_[type]->getPosition(id1);
  hDigiOcc_[type]->Fill(global.x(), global.y());
  hDigiLng_[type]->Fill(global.z());
  hDigiADC_[type]->Fill(adc);
}

void HGCalTBAnalyzer::analyzeRecHits(int type, edm::Handle<HGCRecHitCollection>& hits) {
  std::map<int, double> map_hitLayer;
  std::map<int, std::pair<DetId, double>> map_hitCell;
  for (const auto& it : *hits) {
    DetId detId = it.id();
    GlobalPoint global = hgeom_[type]->getPosition(detId);
    double energy = it.energy();
    int layer = HGCalDetId(detId).layer();
    int cell = HGCalDetId(detId).cell();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "Layer thickness " << hgcons_[type]->waferTypeL(HGCalDetId(detId).wafer());
#endif
    hRecHitOcc_[type]->Fill(global.x(), global.y(), energy);
    hRecHitE_[type]->Fill(energy);
    if (map_hitLayer.count(layer) != 0) {
      map_hitLayer[layer] += energy;
    } else {
      map_hitLayer[layer] = energy;
    }
    if (map_hitCell.count(cell) != 0) {
      double ee = energy + map_hitCell[cell].second;
      map_hitCell[cell] = std::pair<uint32_t, double>(detId, ee);
    } else {
      map_hitCell[cell] = std::pair<uint32_t, double>(detId, energy);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "RecHit: " << layer << " " << global.x() << " " << global.y() << " " << global.z()
                               << " " << energy;
#endif
  }

  for (const auto& itr : map_hitLayer) {
    int layer = itr.first;
    double energy = itr.second;
    double zp = hgcons_[type]->waferZ(layer, true);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCSim") << "SimHit:Layer " << layer << " " << zp << " " << energy;
#endif
    hRecHitLng_[type]->Fill(zp, energy);
    hRecHitLng1_[type]->Fill(layer, energy);
  }

  for (const auto& itr : map_hitCell) {
    DetId detId = ((itr.second).first);
    double energy = ((itr.second).second);
    GlobalPoint global = hgeom_[type]->getPosition(detId);
    hRecHitLat_[type]->Fill(global.x(), global.y(), energy);
  }
}

void HGCalTBAnalyzer::analyzePassiveHits(edm::Handle<edm::PassiveHitContainer> const& hgcPH, int subdet) {
  for (const auto& v : *hgcPH) {
    double energy = v.energy();
    std::string name = v.vname();
    unsigned int id = v.id();
#ifdef EDM_ML_DEBUG
    double time = v.time();
    edm::LogVerbatim("HGCSim") << "HGCalTBAnalyzer::analyzePassiveHits:Energy:"
                               << "Time:Name:Id : " << energy << ":" << time << ":" << name << ":" << id;
#endif

    if (subdet == 1) {
      hgcPassiveEEEnergy_.push_back(energy);
      hgcPassiveEEName_.push_back(name);
      hgcPassiveEEID_.push_back(id);
    } else if (subdet == 2) {
      hgcPassiveFHEnergy_.push_back(energy);
      hgcPassiveFHName_.push_back(name);
      hgcPassiveFHID_.push_back(id);
    } else if (subdet == 3) {
      hgcPassiveBHEnergy_.push_back(energy);
      hgcPassiveBHName_.push_back(name);
      hgcPassiveBHID_.push_back(id);
    } else if (subdet == 4) {
      hgcPassiveCMSEEnergy_.push_back(energy);
      hgcPassiveCMSEName_.push_back(name);
      hgcPassiveCMSEID_.push_back(id);
    } else if (subdet == 5) {
      hgcPassiveBeamEnergy_.push_back(energy);
      hgcPassiveBeamName_.push_back(name);
      hgcPassiveBeamID_.push_back(id);
    }
  }
}

bool HGCalTBAnalyzer::sortTime(const std::pair<double, double>& i, const std::pair<double, double>& j) {
  return i.first < j.first;
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalTBAnalyzer);
