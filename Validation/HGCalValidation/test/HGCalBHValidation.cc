// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

//#define EDM_ML_DEBUG

class HGCalBHValidation: public edm::one::EDAnalyzer<edm::one::WatchRuns,edm::one::SharedResources> {

public:
  explicit HGCalBHValidation(const edm::ParameterSet& ps);
  ~HGCalBHValidation();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  virtual void beginJob() override {}
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override {}
  virtual void analyze(edm::Event const&, edm::EventSetup const&) override;
  void analyzeDigi(HcalDetId const& , double const& , unsigned int &);
private:

  edm::Service<TFileService>                  fs_;
  std::string                                 g4Label_, hcalHits_;
  edm::InputTag                               hcalDigis_;
  int                                         iSample_;
  double                                      threshold_;
  edm::EDGetTokenT<edm::PCaloHitContainer>    tok_hits_;
  edm::EDGetToken                             tok_hbhe_;
  int                                         etaMax_;
  bool                                        ifHCAL_;

  TH1D                                       *hsimE1_, *hsimE2_, *hsimTm_;
  TH1D                                       *hsimLn_, *hdigEn_, *hdigLn_;
  TH2D                                       *hsimOc_, *hsi2Oc_, *hsi3Oc_;
  TH2D                                       *hdigOc_, *hdi2Oc_, *hdi3Oc_;
};


HGCalBHValidation::HGCalBHValidation(const edm::ParameterSet& ps) {

  usesResource("TFileService");

  g4Label_  = ps.getUntrackedParameter<std::string>("ModuleLabel","g4SimHits");
  hcalHits_ = ps.getUntrackedParameter<std::string>("HitCollection","HcalHits");
  hcalDigis_= ps.getUntrackedParameter<edm::InputTag>("DigiCollection");
  iSample_  = ps.getUntrackedParameter<int>("Sample",5);
  threshold_= ps.getUntrackedParameter<double>("Threshold",12.0);
  ifHCAL_   = ps.getUntrackedParameter<bool>("ifHCAL",false);

  tok_hits_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_,hcalHits_));
  if (ifHCAL_) 
    tok_hbhe_ = consumes<QIE11DigiCollection>(hcalDigis_);
  else
    tok_hbhe_ = consumes<HGCBHDigiCollection>(hcalDigis_);
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalBHValidation::Input for SimHit: " 
	    << edm::InputTag(g4Label_,hcalHits_) << "  Digits: " 
	    << hcalDigis_ << "  Sample: " << iSample_ << "  Threshold "
	    << threshold_ << std::endl;
#endif
}

HGCalBHValidation::~HGCalBHValidation() {}

void HGCalBHValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HGCalBHValidation::beginRun(edm::Run const&, edm::EventSetup const& es) {
  
  std::string                        label;
  edm::ESHandle<HcalParameters>      parHandle;
  es.get<HcalParametersRcd>().get(label, parHandle);
  const HcalParameters* hpar = &(*parHandle);
  const std::vector<int> etaM = hpar->etaMax;
  etaMax_ = etaM[1];
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalBHValidation::Maximum Number of eta sectors:"<< etaMax_
	    << "\nHitsValidationHcal::Booking the Histograms" << std::endl;
#endif  

  //Histograms for Sim Hits
  hsimE1_ = fs_->make<TH1D>("SimHitEn1",   "Sim Hit Energy",1000,0.0,1.0);
  hsimE2_ = fs_->make<TH1D>("SimHitEn2",   "Sim Hit Energy",1000,0.0,1.0);
  hsimTm_ = fs_->make<TH1D>("SimHitTime",  "Sim Hit Time",  1000,0.0,500.0);
  hsimLn_ = fs_->make<TH1D>("SimHitLong",  "Sim Hit Long. Profile",40,0.0,20.0);
  hsimOc_ = fs_->make<TH2D>("SimHitOccup", "Sim Hit Occupnacy",2*etaMax_+1,-etaMax_,etaMax_+1,360,0,360);
  hsi2Oc_ = fs_->make<TH2D>("SimHitOccu2", "Sim Hit Occupnacy",2*etaMax_+1,-etaMax_,etaMax_+1,360,0,360);
  hsi3Oc_ = fs_->make<TH2D>("SimHitOccu3", "Sim Hit Occupnacy",2*etaMax_+1,-etaMax_,etaMax_+1,40,0,20);
  //Histograms for Digis
  hdigEn_ = fs_->make<TH1D>("DigiEnergy","Digi ADC Sample",1000,0.0,1000.0);
  hdigLn_ = fs_->make<TH1D>("DigiLong",  "Digi Long. Profile",40,0.0,20.0);
  hdigOc_ = fs_->make<TH2D>("DigiOccup", "Digi Occupnacy",2*etaMax_+1,-etaMax_,etaMax_+1,360,0,360);
  hdi2Oc_ = fs_->make<TH2D>("DigiOccu2", "Digi Occupnacy",2*etaMax_+1,-etaMax_,etaMax_+1,360,0,360);
  hdi3Oc_ = fs_->make<TH2D>("DigiOccu3", "Digi Occupnacy",2*etaMax_+1,-etaMax_,etaMax_+1,40,0,20);
}

void HGCalBHValidation::analyze(const edm::Event& e, const edm::EventSetup& ) {
  
#ifdef EDM_ML_DEBUG  
  std::cout << "HGCalBHValidation:Run = " << e.id().run() << " Event = "
	    << e.id().event();
#endif

  //SimHits
  edm::Handle<edm::PCaloHitContainer> hitsHcal;
  e.getByToken(tok_hits_,hitsHcal); 
#ifdef EDM_ML_DEBUG  
  std::cout << "HGCalBHValidation.: PCaloHitContainer obtained with flag "
	    << hitsHcal.isValid() << std::endl;
#endif
  if (hitsHcal.isValid()) {
#ifdef EDM_ML_DEBUG 
    std::cout << "HGCalBHValidation: PCaloHit buffer " << hitsHcal->size()
	      << std::endl;
    unsigned i(0);
#endif
    std::map<unsigned int,double> map_try;
    for (edm::PCaloHitContainer::const_iterator it = hitsHcal->begin();
	 it != hitsHcal->end(); ++it) {
      double energy    = it->energy();
      double time      = it->time();
      unsigned int id  = it->id();
      int subdet, z, depth, eta, phi, lay;
      HcalTestNumbering::unpackHcalIndex(id, subdet, z, depth, eta, phi, lay);
      if (z==0) eta = -eta;
      if ((subdet == static_cast<int>(HcalEndcap)) ||
	  (subdet == static_cast<int>(HcalBarrel))) 
	hsi2Oc_->Fill((eta+0.1),(phi-0.1),energy);
      if (subdet == static_cast<int>(HcalEndcap)) {
	hsimE1_->Fill(energy);
	hsimTm_->Fill(time,energy);
	hsimOc_->Fill((eta+0.1),(phi-0.1),energy);
	hsimLn_->Fill(lay,energy);
	hsi3Oc_->Fill((eta+0.1),lay,energy);
	double ensum(0);
	if (map_try.count(id) != 0) ensum =  map_try[id];
	ensum += energy;
	map_try[id] = ensum;
#ifdef EDM_ML_DEBUG
	++i;
	std::cout << "HGCalBHHit[" << i << "] ID " << std::hex << " " << id 
		  << std::dec << " SubDet " << subdet << " depth " << depth
		  << " Eta " << eta << " Phi " << phi << " layer " << lay 
		  << " E " << energy << " time " << time << std::endl;
#endif
      }
    }
    for (std::map<unsigned int,double>::iterator itr=map_try.begin();
	 itr != map_try.end(); ++itr) {
      hsimE2_->Fill((*itr).second);
    }
  }

  //Digits
  unsigned int kount(0);
  if (ifHCAL_) {
    edm::Handle<QIE11DigiCollection> hbhecoll;
    e.getByToken(tok_hbhe_, hbhecoll);
#ifdef EDM_ML_DEBUG  
    std::cout << "HGCalBHValidation.: HBHEQIE11DigiCollection obtained with"
	      << " flag " << hbhecoll.isValid() << std::endl;
#endif
    if (hbhecoll.isValid()) {
#ifdef EDM_ML_DEBUG 
      std::cout << "HGCalBHValidation: HBHEDigit buffer " << hbhecoll->size()
		<< std::endl;
#endif
      for (QIE11DigiCollection::const_iterator it=hbhecoll->begin(); 
	   it != hbhecoll->end(); ++it) {
	QIE11DataFrame df(*it);
	HcalDetId cell(df.id());
	double energy = df[iSample_].adc();
	analyzeDigi(cell, energy, kount);
      }
    }
  } else {
    edm::Handle<HGCBHDigiCollection> hbhecoll;
    e.getByToken(tok_hbhe_, hbhecoll);
#ifdef EDM_ML_DEBUG  
    std::cout << "HGCalBHValidation.: HGCBHDigiCollection obtained with"
	      << " flag " << hbhecoll.isValid() << std::endl;
#endif
    if (hbhecoll.isValid()) {
#ifdef EDM_ML_DEBUG 
      std::cout << "HGCalBHValidation: HGCBHDigit buffer " << hbhecoll->size()
		<< std::endl;
#endif
      for (HGCBHDigiCollection::const_iterator it=hbhecoll->begin(); 
	   it != hbhecoll->end(); ++it) {
	HGCBHDataFrame df(*it);
	HcalDetId cell(df.id());
	double energy = df[iSample_].data();
	analyzeDigi(cell, energy, kount);
      }
    }
  }
}

void HGCalBHValidation::analyzeDigi(HcalDetId const& cell, double const& energy,
				    unsigned int &kount) {
  if (energy > threshold_) {
    int    eta    = cell.ieta();
    int    phi    = cell.iphi();
    int    depth  = cell.depth();
    if ((cell.subdet() == HcalEndcap) || (cell.subdet() == HcalBarrel)) 
      hdi2Oc_->Fill((eta+0.1),(phi-0.1));
    if (cell.subdet() == HcalEndcap) {
      hdigEn_->Fill(energy);
      hdigOc_->Fill((eta+0.1),(phi-0.1));
      hdigLn_->Fill(depth);
      hdi3Oc_->Fill((eta+0.1),depth);
      ++kount;
#ifdef EDM_ML_DEBUG
      std::cout << "HGCalBHDigit[" << kount << "] ID " << cell << " E " 
		<< energy << ":" << (energy > threshold_) << std::endl;
#endif
    }
  }
}

DEFINE_FWK_MODULE(HGCalBHValidation);

