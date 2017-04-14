// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
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
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimG4CMS/HGCalTestBeam/interface/AHCalDetId.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TTree.h"

#define EDM_ML_DEBUG

class HGCalTBAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HGCalTBAnalyzer(edm::ParameterSet const&);
  ~HGCalTBAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() override ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  void analyzeSimHits(int type, std::vector<PCaloHit>& hits, double zFront);
  void analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk, 
			edm::Handle<edm::SimVertexContainer> const& SimVtx);
  template<class T1> void analyzeDigi(int type, const T1& detId, uint16_t adc);
  void analyzeRecHits(int type, edm::Handle<HGCRecHitCollection> & hits);

  edm::Service<TFileService>                fs_;
  const HGCalDDDConstants                  *hgcons_[2];
  const HGCalGeometry                      *hgeom_[2];
  bool                                      ifEE_, ifFH_, ifBH_;
  bool                                      doTree_, doTreeCell_;
  bool                                      doSimHits_, doDigis_, doRecHits_;
  std::string                               detectorEE_, detectorFH_, detectorBH_;
  double                                    zFrontEE_, zFrontFH_, zFrontBH_;
  int                                       sampleIndex_;
  edm::EDGetTokenT<edm::PCaloHitContainer>  tok_hitsEE_, tok_hitsFH_, tok_hitsBH_;
  edm::EDGetTokenT<edm::SimTrackContainer>  tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;
  edm::EDGetToken                           tok_digiEE_, tok_digiFH_, tok_digiBH_;
  edm::EDGetToken                           tok_hitrEE_, tok_hitrFH_, tok_hitrBH_;
  edm::EDGetTokenT<edm::HepMCProduct>       tok_hepMC_;
  TTree                                    *tree_;
  TH1D                                     *hSimHitE_[3], *hSimHitT_[3];
  TH1D                                     *hDigiADC_[3], *hDigiLng_[2];
  TH1D                                     *hRecHitE_[3], *hSimHitEn_[3], *hBeam_;
  TH2D                                     *hDigiOcc_[3], *hRecHitOcc_[3];
  TProfile                                 *hSimHitLng_[3], *hSimHitLng1_[3];
  TProfile                                 *hSimHitLng2_[3];
  TProfile                                 *hRecHitLng_[3], *hRecHitLng1_[3];
  TProfile2D                               *hSimHitLat_[3], *hRecHitLat_[3];
  std::vector<TH1D*>                        hSimHitLayEn1EE_, hSimHitLayEn2EE_;
  std::vector<TH1D*>                        hSimHitLayEn1FH_, hSimHitLayEn2FH_;
  std::vector<TH1D*>                        hSimHitLayEn1BH_, hSimHitLayEn2BH_;
  std::vector<float>                        simHitLayEn1EE, simHitLayEn2EE;
  std::vector<float>                        simHitLayEn1FH, simHitLayEn2FH;
  std::vector<float>                        simHitLayEn1BH, simHitLayEn2BH;
  std::vector<uint32_t>                     simHitCellIdEE, simHitCellIdFH;
  std::vector<uint32_t>                     simHitCellIdBH;
  std::vector<float>                        simHitCellEnEE, simHitCellEnFH;
  std::vector<float>                        simHitCellEnBH;
  double                                    xBeam, yBeam, zBeam, pBeam;
};

HGCalTBAnalyzer::HGCalTBAnalyzer(const edm::ParameterSet& iConfig) {

  usesResource("TFileService");

  //now do whatever initialization is needed
  detectorEE_  = iConfig.getParameter<std::string>("DetectorEE");
  detectorFH_  = iConfig.getParameter<std::string>("DetectorFH");
  detectorBH_  = iConfig.getParameter<std::string>("DetectorBH");
  ifEE_        = iConfig.getParameter<bool>("UseEE");
  ifFH_        = iConfig.getParameter<bool>("UseFH");
  ifBH_        = iConfig.getParameter<bool>("UseBH");
  zFrontEE_    = iConfig.getParameter<double>("ZFrontEE");
  zFrontFH_    = iConfig.getParameter<double>("ZFrontFH");
  zFrontBH_    = iConfig.getParameter<double>("ZFrontBH");
  doSimHits_   = iConfig.getParameter<bool>("DoSimHits");
  doDigis_     = iConfig.getParameter<bool>("DoDigis");
  sampleIndex_ = iConfig.getParameter<int>("SampleIndex");
  doRecHits_   = iConfig.getParameter<bool>("DoRecHits");
  doTree_      = iConfig.getUntrackedParameter<bool>("DoTree",false);
  doTreeCell_  = iConfig.getUntrackedParameter<bool>("DoTreeCell",false);
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalTBAnalyzer:: SimHits = " << doSimHits_ << " Digis = "
	    << doDigis_ << ":" << sampleIndex_ << " RecHits = " << doRecHits_
	    << " useDets " << ifEE_ << ":" << ifFH_ << ":" << ifBH_ 
	    << " zFront " << zFrontEE_ << ":" << zFrontFH_ << ":" 
	    << zFrontBH_ << std::endl;
#endif

  edm::InputTag tmp0 = iConfig.getParameter<edm::InputTag>("GeneratorSrc");
  tok_hepMC_   = consumes<edm::HepMCProduct>(tmp0);
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalTBAnalyzer:: GeneratorSource = " << tmp0 << std::endl;
#endif
  std::string   tmp1 = iConfig.getParameter<std::string>("CaloHitSrcEE");
  tok_hitsEE_  = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",tmp1));
  tok_simTk_   = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  tok_simVtx_  = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  edm::InputTag tmp2 = iConfig.getParameter<edm::InputTag>("DigiSrcEE");
  tok_digiEE_  = consumes<HGCEEDigiCollection>(tmp2);
  edm::InputTag tmp3 = iConfig.getParameter<edm::InputTag>("RecHitSrcEE");
  tok_hitrEE_  = consumes<HGCRecHitCollection>(tmp3);
#ifdef EDM_ML_DEBUG
  if (ifEE_) {
    std::cout << "HGCalTBAnalyzer:: Detector " << detectorEE_ << " with tags "
	      << tmp1 << ", " << tmp2 << ", " << tmp3 << std::endl;
  }
#endif
  tmp1         = iConfig.getParameter<std::string>("CaloHitSrcFH");
  tok_hitsFH_  = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",tmp1));
  tmp2         = iConfig.getParameter<edm::InputTag>("DigiSrcFH");
  tok_digiFH_  = consumes<HGCHEDigiCollection>(tmp2);
  tmp3         = iConfig.getParameter<edm::InputTag>("RecHitSrcFH");
  tok_hitrFH_  = consumes<HGCRecHitCollection>(tmp3);
#ifdef EDM_ML_DEBUG
  if (ifFH_) {
    std::cout << "HGCalTBAnalyzer:: Detector " << detectorFH_ << " with tags "
	      << tmp1 << ", " << tmp2 << ", " << tmp3 << std::endl;
  }
#endif
  tmp1         = iConfig.getParameter<std::string>("CaloHitSrcBH");
  tok_hitsBH_  = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits",tmp1));
  tmp2         = iConfig.getParameter<edm::InputTag>("DigiSrcBH");
  tok_digiBH_  = consumes<HGCBHDigiCollection>(tmp2);
  tmp3         = iConfig.getParameter<edm::InputTag>("RecHitSrcBH");
  tok_hitrBH_  = consumes<HGCRecHitCollection>(tmp3);
#ifdef EDM_ML_DEBUG
  if (ifBH_) {
    std::cout << "HGCalTBAnalyzer:: Detector " << detectorBH_ << " with tags "
	      << tmp1 << ", " << tmp2 << ", " << tmp3 << std::endl;
  }
#endif
}

HGCalTBAnalyzer::~HGCalTBAnalyzer() {}

void HGCalTBAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HGCalTBAnalyzer::beginJob() {

  char name[40], title[100];
  hBeam_ = fs_->make<TH1D>("BeamP", "Beam Momentum", 1000, 0, 1000.0);
  for (int i=0; i<3; ++i) {
    bool book(ifEE_);
    std::string det(detectorEE_); 
    if (i == 1) {
      book = ifFH_;
      det  = detectorFH_;
    } else if (i == 2) {
      book = ifBH_;
      det  = detectorBH_;
    }

    if (doSimHits_ && book) {
      sprintf (name, "SimHitEn%s", det.c_str());
      sprintf (title,"Sim Hit Energy for %s", det.c_str());
      hSimHitE_[i] = fs_->make<TH1D>(name,title,100000,0.,0.2);
      sprintf (name, "SimHitEnX%s", det.c_str());
      sprintf (title,"Sim Hit Energy for %s", det.c_str());
      hSimHitEn_[i] = fs_->make<TH1D>(name,title,100000,0.,0.2);
      sprintf (name, "SimHitTm%s", det.c_str());
      sprintf (title,"Sim Hit Timing for %s", det.c_str());
      hSimHitT_[i] = fs_->make<TH1D>(name,title,5000,0.,500.0);
      sprintf (name, "SimHitLat%s", det.c_str());
      sprintf (title,"Lateral Shower profile (Sim Hit) for %s", det.c_str());
      hSimHitLat_[i] = fs_->make<TProfile2D>(name,title,100,-100.,100.,100,-100.,100.);
      sprintf (name, "SimHitLng%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Sim Hit) for %s",det.c_str());
      hSimHitLng_[i] = fs_->make<TProfile>(name,title,50,0.,100.);
      sprintf (name, "SimHitLng1%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Layer) for %s",det.c_str());
      hSimHitLng1_[i] = fs_->make<TProfile>(name,title,200,0.,100.);
      sprintf (name, "SimHitLng2%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Layer) for %s",det.c_str());
      hSimHitLng2_[i] = fs_->make<TProfile>(name,title,200,0.,100.);
    }

    if (doDigis_ && book) {
      sprintf (name, "DigiADC%s", det.c_str());
      sprintf (title,"ADC at Digi level for %s", det.c_str());
      hDigiADC_[i] = fs_->make<TH1D>(name,title,100,0.,100.0);
      sprintf (name, "DigiOcc%s", det.c_str());
      sprintf (title,"Occupancy (Digi)for %s", det.c_str());
      hDigiOcc_[i] = fs_->make<TH2D>(name,title,100,-10.,10.,100,-10.,10.);
      sprintf (name, "DigiLng%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Digi) for %s",det.c_str());
      hDigiLng_[i] = fs_->make<TH1D>(name,title,100,0.,10.);
    }

    if (doRecHits_ && book) {
      sprintf (name, "RecHitEn%s", det.c_str());
      sprintf (title,"Rec Hit Energy for %s", det.c_str());
      hRecHitE_[i] = fs_->make<TH1D>(name,title,1000,0.,10.0);
      sprintf (name, "RecHitOcc%s", det.c_str());
      sprintf (title,"Occupancy (Rec Hit)for %s", det.c_str());
      hRecHitOcc_[i] = fs_->make<TH2D>(name,title,100,-10.,10.,100,-10.,10.);
      sprintf (name, "RecHitLat%s", det.c_str());
      sprintf (title,"Lateral Shower profile (Rec Hit) for %s", det.c_str());
      hRecHitLat_[i] = fs_->make<TProfile2D>(name,title,100,-10.,10.,100,-10.,10.);
      sprintf (name, "RecHitLng%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile (Rec Hit) for %s",det.c_str());
      hRecHitLng_[i] = fs_->make<TProfile>(name,title,100,0.,10.);
      sprintf (name, "RecHitLng1%s", det.c_str());
      sprintf (title,"Longitudinal Shower profile vs Layer for %s",det.c_str());
      hRecHitLng1_[i] = fs_->make<TProfile>(name,title,120,0.,60.);
    }
  }
  if (doSimHits_ && doTree_) {
    tree_ = fs_->make<TTree>("HGCTB","SimHitEnergy");
    tree_->Branch("simHitLayEn1EE", &simHitLayEn1EE);
    tree_->Branch("simHitLayEn2EE", &simHitLayEn2EE);
    tree_->Branch("simHitLayEn1FH", &simHitLayEn1FH);
    tree_->Branch("simHitLayEn2FH", &simHitLayEn2FH);
    tree_->Branch("simHitLayEn1BH", &simHitLayEn1BH);
    tree_->Branch("simHitLayEn2BH", &simHitLayEn2BH);
    tree_->Branch("xBeam",         &xBeam,           "xBeam/D");
    tree_->Branch("yBeam",         &yBeam,           "yBeam/D");
    tree_->Branch("zBeam",         &zBeam,           "zBeam/D");
    tree_->Branch("pBeam",         &pBeam,           "pBeam/D");
    if (doTreeCell_) {
      tree_->Branch("simHitCellIdEE", &simHitCellIdEE);
      tree_->Branch("simHitCellEnEE", &simHitCellEnEE);
      tree_->Branch("simHitCellIdFH", &simHitCellIdFH);
      tree_->Branch("simHitCellEnFH", &simHitCellEnFH);
      tree_->Branch("simHitCellIdBH", &simHitCellIdBH);
      tree_->Branch("simHitCellEnBH", &simHitCellEnBH);
    }
  }
}

void HGCalTBAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {

  char name[40], title[100];
  if (ifEE_) {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(detectorEE_, pHGDC);
    hgcons_[0] = &(*pHGDC);
    if (doDigis_ || doRecHits_) {
      edm::ESHandle<HGCalGeometry> geom;
      iSetup.get<IdealGeometryRecord>().get(detectorEE_, geom);
      hgeom_[0] = geom.product();
    } else {
      hgeom_[0] = 0;
    }
    for (unsigned int l=0; l<hgcons_[0]->layers(false); ++l) {
      sprintf (name, "SimHitEnA%d%s", l, detectorEE_.c_str());
      sprintf (title,"Sim Hit Energy in SIM layer %d for %s",l+1,
	       detectorEE_.c_str());
      hSimHitLayEn1EE_.push_back(fs_->make<TH1D>(name,title,100000,0.,0.2));
      if (l%3 == 0) {
	sprintf (name, "SimHitEnB%d%s", (l/3+1), detectorEE_.c_str());
	sprintf (title,"Sim Hit Energy in layer %d for %s",(l/3+1),
		 detectorEE_.c_str());
	hSimHitLayEn2EE_.push_back(fs_->make<TH1D>(name,title,100000,0.,0.2));
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << "HGCalTBAnalyzer::" << detectorEE_ << " defined with "
	      << hgcons_[0]->layers(false) << " layers" << std::endl;
#endif
  } else {
    hgcons_[0] = 0;
    hgeom_[0]  = 0;
  }

  if (ifFH_) {
    edm::ESHandle<HGCalDDDConstants>  pHGDC;
    iSetup.get<IdealGeometryRecord>().get(detectorFH_, pHGDC);
    hgcons_[1] = &(*pHGDC);
    if (doDigis_ || doRecHits_) {
      edm::ESHandle<HGCalGeometry> geom;
      iSetup.get<IdealGeometryRecord>().get(detectorFH_, geom);
      hgeom_[1] = geom.product();
    } else {
      hgeom_[1] = 0;
    }
    for (unsigned int l=0; l<hgcons_[1]->layers(false); ++l) {
      sprintf (name, "SimHitEnA%d%s", l, detectorFH_.c_str());
      sprintf (title,"Sim Hit Energy in layer %d for %s",l+1,
	       detectorFH_.c_str());
      hSimHitLayEn1FH_.push_back(fs_->make<TH1D>(name,title,100000,0.,0.2));
      if (l%3 == 0) {
	sprintf (name, "SimHitEnB%d%s", (l/3+1), detectorFH_.c_str());
	sprintf (title,"Sim Hit Energy in layer %d for %s",(l/3+1),
		 detectorFH_.c_str());
	hSimHitLayEn2EE_.push_back(fs_->make<TH1D>(name,title,100000,0.,0.2));
      }
    }
#ifdef EDM_ML_DEBUG
    std::cout << "HGCalTBAnalyzer::" << detectorFH_ << " defined with "
	      << hgcons_[1]->layers(false) << " layers" << std::endl;
#endif
  } else {
    hgcons_[1] = 0;
    hgeom_[1]  = 0;
  }
}

void HGCalTBAnalyzer::analyze(const edm::Event& iEvent, 
			      const edm::EventSetup& iSetup) {

  //Generator input
  edm::Handle<edm::HepMCProduct> evtMC;
  iEvent.getByToken(tok_hepMC_,evtMC);
  if (!evtMC.isValid()) {
    edm::LogWarning("HGCal") << "no HepMCProduct found";
  } else { 
    const HepMC::GenEvent * myGenEvent = evtMC->GetEvent();
    unsigned int k(0);
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
         p != myGenEvent->particles_end(); ++p, ++k) {
      if (k == 0) hBeam_->Fill((*p)->momentum().rho());
#ifdef EDM_ML_DEBUG
      std::cout << "Particle[" << k << "] with p " << (*p)->momentum().rho() 
		<< " theta " << (*p)->momentum().theta() << " phi "
		<< (*p)->momentum().phi() << std::endl;
#endif
    }
  }

  //Now the Simhits
  if (doSimHits_) {
    edm::Handle<edm::SimTrackContainer>  SimTk;
    iEvent.getByToken(tok_simTk_, SimTk);
    edm::Handle<edm::SimVertexContainer> SimVtx;
    iEvent.getByToken(tok_simVtx_, SimVtx);
    analyzeSimTracks(SimTk, SimVtx);

    simHitLayEn1EE.clear(); simHitLayEn2EE.clear();
    simHitLayEn1FH.clear(); simHitLayEn2FH.clear();
    simHitLayEn1BH.clear(); simHitLayEn2BH.clear();
    simHitCellIdEE.clear(); simHitCellEnEE.clear();
    simHitCellIdFH.clear(); simHitCellEnFH.clear();
    simHitCellIdBH.clear(); simHitCellEnBH.clear();
    edm::Handle<edm::PCaloHitContainer> theCaloHitContainers;
    std::vector<PCaloHit>               caloHits;
    if (ifEE_) {
      for (unsigned int k=0; k<hgcons_[0]->layers(false); ++k) 
	simHitLayEn1EE.push_back(0);
      for (unsigned int k=0; k<hgcons_[0]->layers(true);  ++k) 
	simHitLayEn2EE.push_back(0);
      iEvent.getByToken(tok_hitsEE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
	std::cout << "PcalohitContainer for " << detectorEE_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	caloHits.clear();
	caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
			theCaloHitContainers->end());
	analyzeSimHits(0, caloHits, zFrontEE_);
      } else {
#ifdef EDM_ML_DEBUG
	std::cout << "PCaloHitContainer does not exist for " << detectorEE_ 
		  << " !!!" << std::endl;
#endif
      }
    }
    if (ifFH_) {
      for (unsigned int k=0; k<hgcons_[1]->layers(false); ++k) 
	simHitLayEn1FH.push_back(0);
      for (unsigned int k=0; k<hgcons_[1]->layers(true);  ++k) 
	simHitLayEn2FH.push_back(0);
      iEvent.getByToken(tok_hitsFH_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
	std::cout << "PcalohitContainer for " << detectorFH_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	caloHits.clear();
	caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
			theCaloHitContainers->end());
	analyzeSimHits(1, caloHits, zFrontFH_);
      } else {
#ifdef EDM_ML_DEBUG
	std::cout << "PCaloHitContainer does not exist for " << detectorFH_ 
		  << " !!!" << std::endl;
#endif
      }
    }
    if (ifBH_) {
      for (int k=0; k<AHCalDetId::MaxDepth; ++k) {
	simHitLayEn1BH.push_back(0);
	simHitLayEn2BH.push_back(0);
      }
      iEvent.getByToken(tok_hitsBH_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
	std::cout << "PcalohitContainer for " << detectorBH_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	caloHits.clear();
	caloHits.insert(caloHits.end(), theCaloHitContainers->begin(), 
			theCaloHitContainers->end());
	analyzeSimHits(2, caloHits, zFrontBH_);
      } else {
#ifdef EDM_ML_DEBUG
	std::cout << "PCaloHitContainer does not exist for " << detectorBH_ 
		  << " !!!" << std::endl;
#endif
      }
    }
    if (doTree_) tree_->Fill();
  }

  //Now the Digis
  if (doDigis_) {
    if (ifEE_) {
      edm::Handle<HGCEEDigiCollection> theDigiContainers;
      iEvent.getByToken(tok_digiEE_, theDigiContainers);
      if (theDigiContainers.isValid()) {
#ifdef EDM_ML_DEBUG
	std::cout << "HGCDigiCintainer for " << detectorEE_ << " with " 
		  << theDigiContainers->size() << " element(s)" << std::endl;
#endif
	for (HGCEEDigiCollection::const_iterator it =theDigiContainers->begin();
	     it !=theDigiContainers->end(); ++it) {
	  HGCEEDetId detId     = (it->id());
	  HGCSample  hgcSample = it->sample(sampleIndex_);
	  uint16_t   adc       = hgcSample.data();
	  analyzeDigi(0, detId, adc);
	}
      }
    }
    if (ifFH_) {
      edm::Handle<HGCHEDigiCollection> theDigiContainers;
      iEvent.getByToken(tok_digiFH_, theDigiContainers);
      if (theDigiContainers.isValid()) {
#ifdef EDM_ML_DEBUG
	std::cout << "HGCDigiContainer for " << detectorFH_ << " with " 
		  << theDigiContainers->size() << " element(s)" << std::endl;
#endif
	for (HGCHEDigiCollection::const_iterator it =theDigiContainers->begin();
	     it !=theDigiContainers->end(); ++it) {
	  HGCHEDetId detId     = (it->id());
	  HGCSample  hgcSample = it->sample(sampleIndex_);
	  uint16_t   adc       = hgcSample.data();
	  analyzeDigi(1, detId, adc);
	}
      }
    }
  }

  //The Rechits
  if (doRecHits_) {
    edm::Handle<HGCRecHitCollection> theCaloHitContainers;
    if (ifEE_) {
      iEvent.getByToken(tok_hitrEE_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
	std::cout << "HGCRecHitCollection for " << detectorEE_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	analyzeRecHits(0, theCaloHitContainers);
      } else {
#ifdef EDM_ML_DEBUG
	std::cout << "HGCRecHitCollection does not exist for " << detectorEE_ 
		  << " !!!" << std::endl;
#endif
      }
    }
    if (ifFH_) {
      iEvent.getByToken(tok_hitrFH_, theCaloHitContainers);
      if (theCaloHitContainers.isValid()) {
#ifdef EDM_ML_DEBUG
	std::cout << "HGCRecHitCollection for " << detectorFH_ << " has "
		  << theCaloHitContainers->size() << " hits" << std::endl;
#endif
	analyzeRecHits(1, theCaloHitContainers);
      } else {
#ifdef EDM_ML_DEBUG
	std::cout << "HGCRecHitCollection does not exist for " << detectorFH_ 
		  << " !!!" << std::endl;
#endif
      }
    }
  }
}

void HGCalTBAnalyzer::analyzeSimHits (int type, std::vector<PCaloHit>& hits,
				      double zFront) {

  std::map<uint32_t,double>                 map_hits, map_hitn;
  std::map<int,double>                      map_hitDepth;
  std::map<int,std::pair<uint32_t,double> > map_hitLayer, map_hitCell;
  double                                    entot(0);
  for (unsigned int i=0; i<hits.size(); i++) {
    double energy      = hits[i].energy();
    double time        = hits[i].time();
    uint32_t id        = hits[i].id();
    entot             += energy;
    int      subdet, zside, layer, sector, subsector(0), cell, depth(0), idx(0);
    if (type == 2) {
      subdet           = HcalDetId(id).subdet();
      if (subdet != HcalOther) continue;
      AHCalDetId hid(id);
      layer = depth    = hid.depth();
      zside            = hid.zside();
      sector           = hid.irow();
      cell             = hid.icol();
      idx              = ((hid.irowAbs()*100) + (hid.icolAbs()));
    } else {
      HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector,
					     subsector, cell);
      depth           = hgcons_[type]->simToReco(cell,layer,sector,true).second;
      idx             = sector*1000+cell;
    }
#ifdef EDM_ML_DEBUG
    std::cout << "SimHit:Hit[" << i << "] Id " << subdet << ":" << zside << ":"
	      << layer << ":" << sector << ":" << subsector << ":" << cell 
	      << ":" << depth << " Energy " << energy << " Time " << time
	      << std::endl;
#endif
    if (map_hits.count(id) != 0) {
      map_hits[id] += energy;
    } else {
      map_hits[id]  = energy;
    }
    if (map_hitLayer.count(layer) != 0) {
      double ee           = energy + map_hitLayer[layer].second;
      map_hitLayer[layer] = std::pair<uint32_t,double>(id,ee);
    } else {
      map_hitLayer[layer] = std::pair<uint32_t,double>(id,energy);
    }
    if (depth >= 0) {
      if (map_hitCell.count(idx) != 0) {
	double ee        = energy + map_hitCell[idx].second;
	map_hitCell[idx] = std::pair<uint32_t,double>(id,ee);
      } else {
	map_hitCell[idx] = std::pair<uint32_t,double>(id,energy);
      }
      if (map_hitDepth.count(depth) != 0) {
	map_hitDepth[depth] += energy;
      } else {
	map_hitDepth[depth]  = energy;
      }
      uint32_t idn  = (type == 2) ? id :
	HGCalTestNumbering::packHexagonIndex(subdet, zside, depth, sector, 
					     subsector, cell);
      if (map_hitn.count(idn) != 0) {
	map_hitn[idn] += energy;
      } else {
	map_hitn[idn]  = energy;
      }
    }
    hSimHitT_[type]->Fill(time,energy);
  }

  hSimHitEn_[type]->Fill(entot);
  for (std::map<uint32_t,double>::iterator itr = map_hits.begin() ; 
       itr != map_hits.end(); ++itr)   {
    hSimHitE_[type]->Fill(itr->second);
  }

  for (std::map<int,std::pair<uint32_t,double> >::iterator itr = map_hitLayer.begin(); 
       itr != map_hitLayer.end(); ++itr)   {
    int    layer      = itr->first - 1;
    double energy     = (itr->second).second;
    double zp(0);
    if (type != 2) zp = hgcons_[type]->waferZ(layer+1,false);
    else           zp = AHCalDetId((itr->second).first).getZ();
#ifdef EDM_ML_DEBUG
    std::cout << "SimHit:Layer " << layer+1 << " Z " << zp << ":" << zp-zFront
	      << " E " << energy << std::endl;
#endif
    hSimHitLng_[type]->Fill(zp-zFront,energy);
    hSimHitLng2_[type]->Fill(layer+1,energy);
    if (type == 0) {
      if (layer < (int)(hSimHitLayEn1EE_.size())) {
	simHitLayEn1EE[layer] = energy;
	hSimHitLayEn1EE_[layer]->Fill(energy);
      }
    } else {
      if (layer < (int)(hSimHitLayEn1FH_.size())) {
	simHitLayEn1FH[layer] = energy;
	hSimHitLayEn1FH_[layer]->Fill(energy);
      }
    }
  }
  for (std::map<int,double>::iterator itr = map_hitDepth.begin(); 
       itr != map_hitDepth.end(); ++itr)   {
    int    layer      = itr->first - 1;
    double energy     = itr->second;
#ifdef EDM_ML_DEBUG
    std::cout << "SimHit:Layer " << layer+1 << " " << energy << std::endl;
#endif
    hSimHitLng1_[type]->Fill(layer+1,energy);
    if (type == 0) {
      if (layer < (int)(hSimHitLayEn2EE_.size())) {
	simHitLayEn2EE[layer] = energy;
	hSimHitLayEn2EE_[layer]->Fill(energy);
      }
    } else if (type == 1) {
      if (layer < (int)(hSimHitLayEn2FH_.size())) {
	simHitLayEn2FH[layer] = energy;
	hSimHitLayEn2FH_[layer]->Fill(energy);
      }
    } else if (type == 2) {
      if (layer < (int)(hSimHitLayEn2BH_.size())) {
	simHitLayEn2BH[layer] = energy;
	hSimHitLayEn2BH_[layer]->Fill(energy);
      }
    }
  }

  for (std::map<int,std::pair<uint32_t,double> >::iterator itr = map_hitCell.begin(); 
       itr != map_hitCell.end(); ++itr) {
    uint32_t id       = ((itr->second).first);
    double   energy   = ((itr->second).second);
    std::pair<float,float> xy(0,0);
    double xx(0);
    if (type == 2) {
      xy = AHCalDetId(id).getXY();
      xx = xy.first;
    } else {
      int      subdet, zside, layer, sector, subsector, cell;
      HGCalTestNumbering::unpackHexagonIndex(id, subdet, zside, layer, sector,
					     subsector, cell);
      xy        = hgcons_[type]->locateCell(cell,layer,sector,false);
      double zp = hgcons_[type]->waferZ(layer,false);
      xx        = (zp < 0) ? -xy.first : xy.first;
    }
    hSimHitLat_[type]->Fill(xx,xy.second,energy);
  }

  for (std::map<uint32_t,double>::iterator itr = map_hitn.begin(); 
       itr != map_hitn.end(); ++itr) {
    uint32_t id     = itr->first;
    double   energy = itr->second;
    if (type == 0) {
      simHitCellIdEE.push_back(id); simHitCellEnEE.push_back(energy);
    } else if (type == 1) {
      simHitCellIdFH.push_back(id); simHitCellEnFH.push_back(energy);
    } else if (type == 2) {
      simHitCellIdBH.push_back(id); simHitCellEnBH.push_back(energy);
    }
  }
}

void HGCalTBAnalyzer::analyzeSimTracks(edm::Handle<edm::SimTrackContainer> const& SimTk, 
				       edm::Handle<edm::SimVertexContainer> const& SimVtx) {

  xBeam = yBeam = zBeam = pBeam = -1000000;
  int vertIndex(-1);
  for (edm::SimTrackContainer::const_iterator simTrkItr = SimTk->begin(); 
       simTrkItr!= SimTk->end(); simTrkItr++) {
#ifdef EDM_ML_DEBUG
    std::cout << "Track " << simTrkItr->trackId() << " Vertex "
	      << simTrkItr->vertIndex() << " Type " << simTrkItr->type()
	      << " Charge " << simTrkItr->charge() << " momentum "
	      << simTrkItr->momentum() << " " << simTrkItr->momentum().P()
	      << std::endl;
#endif
    if (vertIndex == -1) {
      vertIndex = simTrkItr->vertIndex();
      pBeam     = simTrkItr->momentum().P();
    }
  }
  if (vertIndex != -1 && vertIndex < (int)SimVtx->size()) {
    edm::SimVertexContainer::const_iterator simVtxItr= SimVtx->begin();
    for (int iv=0; iv<vertIndex; iv++) simVtxItr++;
#ifdef EDM_ML_DEBUG
    std::cout << "Vertex " << vertIndex << " position "
	      << simVtxItr->position() << std::endl;
#endif
    xBeam = simVtxItr->position().X();
    yBeam = simVtxItr->position().Y();
    zBeam = simVtxItr->position().Z();
  }

}

template<class T1>
void HGCalTBAnalyzer::analyzeDigi (int type, const T1& detId, uint16_t adc) {

  DetId id1 = DetId(detId.rawId());
  GlobalPoint global = hgeom_[type]->getPosition(id1);
  hDigiOcc_[type]->Fill(global.x(),global.y());
  hDigiLng_[type]->Fill(global.z());
  hDigiADC_[type]->Fill(adc);
}

void HGCalTBAnalyzer::analyzeRecHits (int type, 
				      edm::Handle<HGCRecHitCollection>& hits) {
 
  std::map<int,double>                   map_hitLayer;
  std::map<int,std::pair<DetId,double> > map_hitCell;
  for (HGCRecHitCollection::const_iterator it = hits->begin(); 
       it != hits->end(); ++it) {
    DetId       detId  = it->id();
    GlobalPoint global = hgeom_[type]->getPosition(detId);
    double      energy = it->energy();
    int         layer  = HGCalDetId(detId).layer();
    int         cell   = HGCalDetId(detId).cell();
    hRecHitOcc_[type]->Fill(global.x(),global.y(),energy);
    hRecHitE_[type]->Fill(energy);
    if (map_hitLayer.count(layer) != 0) {
      map_hitLayer[layer] += energy;
    } else {
      map_hitLayer[layer]  = energy;
    }
    if (map_hitCell.count(cell) != 0) {
      double ee         = energy + map_hitCell[cell].second;
      map_hitCell[cell] = std::pair<uint32_t,double>(detId,ee);
    } else {
      map_hitCell[cell] = std::pair<uint32_t,double>(detId,energy);
    }
#ifdef EDM_ML_DEBUG
    std::cout << "RecHit: " << layer  << " " << global.x() << " " << global.y()
	      << " " << global.z() << " " << energy << std::endl;
#endif
  }

  for (std::map<int,double>::iterator itr = map_hitLayer.begin(); 
       itr != map_hitLayer.end(); ++itr)   {
    int    layer      = itr->first;
    double energy     = itr->second;
    double zp         = hgcons_[type]->waferZ(layer,true);
#ifdef EDM_ML_DEBUG
    std::cout << "SimHit:Layer " << layer << " " << zp << " " << energy 
	      << std::endl;
#endif
    hRecHitLng_[type]->Fill(zp,energy);
    hRecHitLng1_[type]->Fill(layer,energy);
  }

  for (std::map<int,std::pair<DetId,double> >::iterator itr = map_hitCell.begin(); 
       itr != map_hitCell.end(); ++itr) {
    DetId       detId  = ((itr->second).first);
    double      energy = ((itr->second).second);
    GlobalPoint global = hgeom_[type]->getPosition(detId);
    hRecHitLat_[type]->Fill(global.x(),global.y(),energy);
  }
}
  
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTBAnalyzer);
