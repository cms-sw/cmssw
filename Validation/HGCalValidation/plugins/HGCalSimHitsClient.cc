#include "Validation/HGCalValidation/plugins/HGCalSimHitsClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include"DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

HGCalSimHitsClient::HGCalSimHitsClient(const edm::ParameterSet& iConfig) {

  dbe_ = edm::Service<DQMStore>().operator->();
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  nameDetector_  = iConfig.getParameter<std::string>("DetectorName");
  verbosity_     = iConfig.getUntrackedParameter<int>("Verbosity",0);

  if (!dbe_) {
    edm::LogError("HGCalSimHitsClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if (iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if (dbe_) dbe_->setVerbose(0);
  }

  //dirName_= iConfig.getParameter<std::string>("DQMDirName");
  if (dbe_) dbe_->setCurrentFolder("/");
}

HGCalSimHitsClient::~HGCalSimHitsClient() { }

void HGCalSimHitsClient::beginJob() { 
  geometrydefined_ = false;
  symmDet_         = true;
}

void HGCalSimHitsClient::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void HGCalSimHitsClient::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) { 
  if (!geometrydefined_) {
    edm::ESTransientHandle<DDCompactView> pDD;
    iSetup.get<IdealGeometryRecord>().get( pDD );
    geometrydefined_ = defineGeometry(pDD);
  }
}

void HGCalSimHitsClient::endRun(const edm::Run& , const edm::EventSetup& ) {
  runClient_();
}

//dummy analysis function
void HGCalSimHitsClient::analyze(const edm::Event& , const edm::EventSetup&) { }
void HGCalSimHitsClient::endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& ) { }

void HGCalSimHitsClient::runClient_() {
  if (!dbe_) return; 
  dbe_->setCurrentFolder("/"); 
  if (verbosity_>0) std::cout << "\nrunClient" << std::endl;
  std::vector<MonitorElement*> hgcalMEs;
  std::vector<std::string> fullDirPath = dbe_->getSubdirs();

  for (unsigned int i=0; i<fullDirPath.size(); i++) {
    if (verbosity_>0) std::cout << "\nfullPath: " << fullDirPath.at(i) << std::endl;
    dbe_->setCurrentFolder(fullDirPath.at(i));
    std::vector<std::string> fullSubDirPath = dbe_->getSubdirs();

    for (unsigned int j=0; j<fullSubDirPath.size(); j++) {
      if (verbosity_>0) std::cout << "fullSubPath: " << fullSubDirPath.at(j) << std::endl;
      std::string nameDirectory = "HGCalSimHitsV/"+nameDetector_;

      if (strcmp(fullSubDirPath.at(j).c_str(), nameDirectory.c_str()) == 0) {
        hgcalMEs = dbe_->getContents(fullSubDirPath.at(j));
	if (verbosity_>0) std::cout << "hgcalMES size : " << hgcalMEs.size() <<std::endl;
        if( !SimHitsEndjob(hgcalMEs) ) std::cout<< "\nError in SimhitsEndjob!" << std::endl <<std::endl;
      }
    }
  }
}

int HGCalSimHitsClient::SimHitsEndjob(const std::vector<MonitorElement*>& hgcalMEs) {
  if(!geometrydefined_) return 0;
  std::vector<MonitorElement*> energy_[6];
  std::vector<MonitorElement*> EtaPhi_Plus_;
  std::vector<MonitorElement*> EtaPhi_Minus_;
  std::vector<MonitorElement*> HitOccupancy_Plus_[4];
  std::vector<MonitorElement*> HitOccupancy_Minus_[4];
  std::vector<MonitorElement*> MeanHitOccupancy_Plus_;
  std::vector<MonitorElement*> MeanHitOccupancy_Minus_;

  std::ostringstream name;
  double nevent;
  int nbinsx, nbinsy;
  unsigned int layers_ = hgcons_->layers(false);
  for (unsigned int ilayer = 0; ilayer < layers_; ilayer++ ){ 
    for (int itimeslice = 0; itimeslice < 6 ; itimeslice++ ) {
      //Energy
      name.str(""); name << "energy_time_" << itimeslice << "_layer_" << ilayer;
      for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
	if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
	  energy_[itimeslice].push_back(hgcalMEs[ih]);
	}
      }
      //normalization
      nevent = energy_[itimeslice].at(ilayer)->getEntries();
      nbinsx = energy_[itimeslice].at(ilayer)->getNbinsX();
      for(int i=1; i <= nbinsx; i++) {
	double binValue = energy_[itimeslice].at(ilayer)->getBinContent(i)/nevent;
	energy_[itimeslice].at(ilayer)->setBinContent(i,binValue);
      }
    }///loop over timeslice
    
    //EtaPhi 2d plots
    name.str(""); name << "EtaPhi_Plus_" << "layer_" << ilayer;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
	EtaPhi_Plus_.push_back(hgcalMEs[ih]);
      }
    }

    name.str(""); name << "EtaPhi_Minus_" << "layer_" << ilayer;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
        EtaPhi_Minus_.push_back(hgcalMEs[ih]);
      }
    }
    //normalization EtaPhi
    nevent = EtaPhi_Plus_.at(ilayer)->getEntries();
    nbinsx = EtaPhi_Plus_.at(ilayer)->getNbinsX();
    nbinsy = EtaPhi_Plus_.at(ilayer)->getNbinsY();
    for(int i=1; i<= nbinsx; ++i) {
      for(int j=1; j<= nbinsy; ++j) {
	double binValue = EtaPhi_Plus_.at(ilayer)->getBinContent(i, j)/nevent;
	EtaPhi_Plus_.at(ilayer)->setBinContent(i, j, binValue);
      }
    }

    nevent =  EtaPhi_Minus_.at(ilayer)->getEntries();
    nbinsx =  EtaPhi_Minus_.at(ilayer)->getNbinsX();
    nbinsy = EtaPhi_Plus_.at(ilayer)->getNbinsY();
    for(int i=1; i<= nbinsx; ++i) {
      for(int j=1; j<= nbinsy; ++j) {
        double binValue =  EtaPhi_Minus_.at(ilayer)->getBinContent(i, j)/nevent;
         EtaPhi_Minus_.at(ilayer)->setBinContent(i, j, binValue);
      }
    }

    //HitOccupancy
    for(int indx=0; indx<4; ++indx){
      name.str(""); name << "HitOccupancy_Plus"<< indx << "_layer_" << ilayer;
      for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
	if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
	  HitOccupancy_Plus_[indx].push_back(hgcalMEs[ih]);
	}
      }

      name.str(""); name << "HitOccupancy_Minus"<< indx << "_layer_" << ilayer;
      for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
        if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
          HitOccupancy_Minus_[indx].push_back(hgcalMEs[ih]);
	}
      }
    }
    for(int indx=0; indx<4; ++indx){
      nevent = HitOccupancy_Plus_[indx].at(ilayer)->getEntries();
      nbinsx = HitOccupancy_Plus_[indx].at(ilayer)->getNbinsX();
      for(int i=1; i<= nbinsx; ++i) {
	double binValue = HitOccupancy_Plus_[indx].at(ilayer)->getBinContent(i)/nevent;
	HitOccupancy_Plus_[indx].at(ilayer)->setBinContent(i, binValue);
      }

      nevent = HitOccupancy_Minus_[indx].at(ilayer)->getEntries();
      nbinsx = HitOccupancy_Minus_[indx].at(ilayer)->getNbinsX();
      for(int i=1; i<= nbinsx; ++i) {
        double binValue = HitOccupancy_Minus_[indx].at(ilayer)->getBinContent(i)/nevent;
        HitOccupancy_Minus_[indx].at(ilayer)->setBinContent(i, binValue);
      }

    }

  }//loop over layers

  for(int indx=0; indx<4; ++indx) {
    name.str(""); name << "MeanHitOccupancy_Plus"<< indx ;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
	MeanHitOccupancy_Plus_.push_back(hgcalMEs[ih]);
      }
    }

    name.str(""); name << "MeanHitOccupancy_Minus"<< indx ;
    for(unsigned int ih=0; ih<hgcalMEs.size(); ih++){
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0){
        MeanHitOccupancy_Minus_.push_back(hgcalMEs[ih]);
      }
    }
  }

  return 1;
}

bool HGCalSimHitsClient::defineGeometry(edm::ESTransientHandle<DDCompactView> &ddViewH){
  const DDCompactView & cview = *ddViewH;
  hgcons_ = new HGCalDDDConstants(cview, nameDetector_);
  if (verbosity_>0) std::cout << "Initialize HGCalDDDConstants for " 
                              << nameDetector_ << " : " << hgcons_ <<std::endl;

  std::string attribute = "Volume"; 
  std::string value     = nameDetector_;
  DDValue val(attribute, value, 0);

  DDSpecificsFilter filter;
  filter.setCriteria(val, DDSpecificsFilter::equals);
  DDFilteredView fv(cview);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();

  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    std::string name = sol.name();
    int isd = (name.find(nameDetector_) == std::string::npos) ? -1 : 1;
    if (isd > 0) {
      std::vector<int> copy = fv.copyNumbers();
      int nsiz = (int)(copy.size());
      int lay  = (nsiz > 0) ? copy[nsiz-1] : -1;
      int sec  = (nsiz > 1) ? copy[nsiz-2] : -1;
      int zp   = (nsiz > 3) ? copy[nsiz-4] : -1;
      if (zp !=1 ) zp = -1;
      const DDTrap & trp = static_cast<DDTrap>(sol);
      int subs = (trp.alpha1()>0 ? 1 : 0);
      symmDet_ = (trp.alpha1()==0 ? true : false);
      uint32_t id = HGCalDetId(ForwardEmpty,zp,lay,sec,subs,0).rawId();
      DD3Vector x, y, z;
      fv.rotation().GetComponents( x, y, z ) ;
      const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
                                        x.Y(), y.Y(), z.Y(),
                                        x.Z(), y.Z(), z.Z() );
      const CLHEP::HepRotation hr ( rotation );
      const CLHEP::Hep3Vector h3v ( fv.translation().X(),
                                    fv.translation().Y(),
                                    fv.translation().Z()  ) ;
      const HepGeom::Transform3D ht3d (hr, h3v);
      transMap_.insert(std::make_pair(id,ht3d));
      if (verbosity_>2) std::cout << HGCalDetId(id) << " Transform using " 
                                  << h3v << " and " << hr;
    }
    dodet = fv.next();
  }
  if (verbosity_>0) std::cout << "Finds " << transMap_.size() << " elements" 
                              << " and SymmDet_ = " << symmDet_ << std::endl;
  return true;
}

DEFINE_FWK_MODULE(HGCalSimHitsClient);
