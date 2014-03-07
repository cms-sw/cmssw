#include "Validation/HcalHits/interface/HcalSimHitsClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"


HcalSimHitsClient::HcalSimHitsClient(const edm::ParameterSet& iConfig):conf_(iConfig) {
  
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");
//  maxDepthHB_ = iConfig.getUntrackedParameter<int>("MaxDepthHB", 3);
//  maxDepthHE_ = iConfig.getUntrackedParameter<int>("MaxDepthHE", 5);
//  maxDepthHO_ = iConfig.getUntrackedParameter<int>("MaxDepthHO", 4);
//  maxDepthHF_ = iConfig.getUntrackedParameter<int>("MaxDepthHF", 2);

  
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("HcalSimHitsClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if (iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if (dbe_) dbe_->setVerbose(0);
  }
  
  debug_ = false;
  verbose_ = true;
  
  dirName_= iConfig.getParameter<std::string>("DQMDirName");
  if (dbe_) dbe_->setCurrentFolder(dirName_);
  
}


HcalSimHitsClient::~HcalSimHitsClient() { }

void HcalSimHitsClient::beginJob() { }

void HcalSimHitsClient::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void HcalSimHitsClient::beginRun(const edm::Run& run, const edm::EventSetup& c) {
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  c.get<HcalRecNumberingRecord>().get( pHRNDC );
  hcons = &(*pHRNDC);
  maxDepthHB_ = hcons->getMaxDepth(0);
  maxDepthHE_ = hcons->getMaxDepth(1);
  maxDepthHF_ = hcons->getMaxDepth(2);
  maxDepthHO_ = hcons->getMaxDepth(3);

  std::cout << " Maximum Depths HB:"<< maxDepthHB_
	    << " HE:" << maxDepthHE_ << " HO:"
	    << maxDepthHO_ << " HF:"<<maxDepthHF_<<std::endl;

  

}



void HcalSimHitsClient::endRun(const edm::Run& , const edm::EventSetup& ) {
  runClient_();
}

//dummy analysis function
void HcalSimHitsClient::analyze(const edm::Event& , const edm::EventSetup&) { }

void HcalSimHitsClient::endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& ) { }

void HcalSimHitsClient::runClient_() {
  
  if (!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);
  
  if (verbose_) std::cout << "\nrunClient" << std::endl; 
  
  std::vector<MonitorElement*> hcalMEs;
  
  std::vector<std::string> fullPathHLTFolders = dbe_->getSubdirs();
  for (unsigned int i=0;i<fullPathHLTFolders.size();i++) {
    if (verbose_) std::cout <<"\nfullPath: "<< fullPathHLTFolders[i] << std::endl;
    dbe_->setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = dbe_->getSubdirs();
    for (unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {
      
      if (verbose_) std::cout <<"fullSub: "<<fullSubPathHLTFolders[j] << std::endl;
      
      if (strcmp(fullSubPathHLTFolders[j].c_str(), "HcalHitsV/SimHitsValidationHcal") == 0) {
	hcalMEs = dbe_->getContents(fullSubPathHLTFolders[j]);
	if (verbose_) std::cout <<"hltMES size : "<<hcalMEs.size()<<std::endl;
	
	if( !SimHitsEndjob(hcalMEs) ) std::cout<<"\nError in SimhitEndjob!"<<std::endl<<std::endl;
      }
      
    }    
    
  }
  
}

// called after entering the  directory
// hcalMEs are within that directory
int HcalSimHitsClient::SimHitsEndjob(const std::vector<MonitorElement*> &hcalMEs) {
  

  std::vector<std::string> divisions = getHistogramTypes();
  MonitorElement *Occupancy_map[nTime][divisions.size()];
  MonitorElement *Energy[nType1], *Time_weighteden[nType1];
  MonitorElement *HitEnergyvsieta[divisions.size()], *HitTimevsieta[divisions.size()];

  
  std::string time[nTime]={"25","50","100","250"};
  std::string detdivision[nType1]={"HB","HE","HF","HO"};
  char name[40], name1[40], name2[40], name3[40], name4[40];
  
  
  for (int k=0; k<nType1;k++) {
      for (unsigned int ih=0; ih<hcalMEs.size(); ih++) {
	sprintf (name1, "Energy_%s", detdivision[k].c_str());
	sprintf (name2, "Time_Enweighted_%s", detdivision[k].c_str());
	if (strcmp(hcalMEs[ih]->getName().c_str(), name1) == 0) {
	  Energy[k] = hcalMEs[ih];
	}
	if (strcmp(hcalMEs[ih]->getName().c_str(), name2) == 0) {
	  Time_weighteden[k] = hcalMEs[ih];
	}
      }
  }
  

  for (int i=0; i<nTime; i++) {
    for (unsigned int j=0; j<divisions.size();j++) {
	for (unsigned int ih=0; ih<hcalMEs.size(); ih++) {
	  sprintf (name, "HcalHitE%s%s", time[i].c_str(),divisions[j].c_str());
	  if (strcmp(hcalMEs[ih]->getName().c_str(), name) == 0) {
	    Occupancy_map[i][j]= hcalMEs[ih];
	  }

	}

    }
  }

    
  for (unsigned int k=0; k<divisions.size();k++) {
    for (unsigned int ih=0; ih<hcalMEs.size(); ih++) {
      sprintf (name3, "HcalHitEta%s",divisions[k].c_str());
      sprintf (name4, "HcalHitTimeAEta%s",divisions[k].c_str());
	if (strcmp(hcalMEs[ih]->getName().c_str(), name3) == 0) {
	  HitEnergyvsieta[k]= hcalMEs[ih];
	}
	if (strcmp(hcalMEs[ih]->getName().c_str(), name4) == 0) {
	  HitTimevsieta[k]= hcalMEs[ih];
	}
    }
  }
  
  
  //mean energy 
  
  double nevent = Energy[0]->getEntries();
  if (verbose_) std::cout<<"nevent : "<<nevent<<std::endl;
  
  float cont[nTime][divisions.size()];
  float en[nType1], tme[nType1];
  float hitenergy[divisions.size()], hittime[divisions.size()];
  float fev = float(nevent);
  
  for(int dettype=0; dettype<nType1; dettype++)
    {
      int nx1=Energy[dettype]->getNbinsX();
      for(int i=0; i<=nx1; i++)
	{
	  en[dettype]= Energy[dettype]->getBinContent(i)/fev;
	  Energy[dettype]->setBinContent(i,en[dettype]);
	}
      int nx2= Time_weighteden[dettype]->getNbinsX();
      for(int i=0; i<=nx2; i++)
	{
	  tme[dettype]= Time_weighteden[dettype]->getBinContent(i)/fev;
	  Time_weighteden[dettype]->setBinContent(i,tme[dettype]);
	}
    }
  
  
  for(unsigned int dettype=0; dettype<divisions.size(); dettype++)
    {
      int nx1=HitEnergyvsieta[dettype]->getNbinsX();
      for(int i=0; i<=nx1; i++)
	{
	  hitenergy[dettype]= HitEnergyvsieta[dettype]->getBinContent(i)/fev;
	  HitEnergyvsieta[dettype]->setBinContent(i,hitenergy[dettype]);
	}
      int nx2= HitTimevsieta[dettype]->getNbinsX();
      for(int i=0; i<=nx2; i++)
	{
	  hittime[dettype]= HitTimevsieta[dettype]->getBinContent(i)/fev;
	  HitTimevsieta[dettype]->setBinContent(i,hittime[dettype]);
	}
    }
  

  
  for (int itime=0; itime<nTime; itime++) {
    for (unsigned int det=0; det<divisions.size();det++) {
      std::cout<<"itime:"<<itime<<"det:"<<det<<std::endl;
      int ny= Occupancy_map[itime][det]->getNbinsY();
      int nx= Occupancy_map[itime][det]->getNbinsX(); 

      for (int i=1; i<nx+1; i++) {
	for (int j=1; j<ny+1; j++) {
	  
	  cont[itime][det] = Occupancy_map[itime][det]->getBinContent(i,j)/fev ;
	  
	  Occupancy_map[itime][det]->setBinContent(i,j,cont[itime][det]);
	}
      }
    }
  }
  

  
  
  
  return 1;
}


std::vector<std::string> HcalSimHitsClient::getHistogramTypes() {

  int maxDepth = std::max(maxDepthHB_,maxDepthHE_);
  std::cout<<"Max depth 1st step::"<<maxDepth<<std::endl;
  maxDepth = std::max(maxDepth,maxDepthHF_);
  std::cout<<"Max depth 2nd step::"<<maxDepth<<std::endl;
  maxDepth = std::max(maxDepth,maxDepthHO_);
  std::cout<<"Max depth 3rd step::"<<maxDepth<<std::endl;
  std::vector<std::string > divisions;
  char                               name1[20];

  //first overall Hcal                                                                                                                                                                                                                                                 
  for (int depth=0; depth<maxDepth; ++depth) {
    sprintf (name1, "HC%d", depth);
    divisions.push_back(std::string(name1));
  }
  //HB                                                                                                                                                                                                                                                                 
  for (int depth=0; depth<maxDepthHB_; ++depth) {
    sprintf (name1, "HB%d", depth);
    divisions.push_back(std::string(name1));
  }
  //HE                                                                                                                                                              
  for (int depth=0; depth<maxDepthHE_; ++depth) {
    sprintf (name1, "HE%d+z", depth);
    divisions.push_back(std::string(name1));
    sprintf (name1, "HE%d-z", depth);
    divisions.push_back(std::string(name1));

  }
  //HO                                                                                                                                                              
  {
    int depth = maxDepthHO_;
    sprintf (name1, "HO%d", depth);
    divisions.push_back(std::string(name1));
  }

  //HF (first absorber, then different types of abnormal hits)                                                                                                                             
  std::string hfty1[4] = {"A","W","B","J"};
  //  int         dept0[4] = {0, 1, 2, 3};
  for (int k=0; k<4; ++k) {
    for (int depth=0; depth<maxDepthHF_; ++depth) {
      sprintf (name1, "HF%s%d+z", hfty1[k].c_str(), depth);
      divisions.push_back(std::string(name1));
      sprintf (name1, "HF%s%d-z", hfty1[k].c_str(), depth);
      divisions.push_back(std::string(name1));
    }
  }
  return divisions;


}



DEFINE_FWK_MODULE(HcalSimHitsClient);
