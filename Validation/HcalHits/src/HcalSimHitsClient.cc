#include "Validation/HcalHits/interface/HcalSimHitsClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalSimHitsClient::HcalSimHitsClient(const edm::ParameterSet& iConfig):conf_(iConfig) {

  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");

 
  debug_ = false;
  verbose_ = false;

  dirName_= iConfig.getParameter<std::string>("DQMDirName");
 
}


HcalSimHitsClient::~HcalSimHitsClient() { }

// let's see if this breaks anything
/*void HcalSimHitsClient::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}*/



void HcalSimHitsClient::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig )
{
  runClient_(ib,ig);
}




void HcalSimHitsClient::runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig) {

  ig.setCurrentFolder(dirName_);
  
  if (verbose_) std::cout << "\nrunClient" << std::endl; 

  std::vector<MonitorElement*> hcalMEs;
  
  std::vector<std::string> fullPathHLTFolders = ig.getSubdirs();
  for (unsigned int i=0;i<fullPathHLTFolders.size();i++) {
    if (verbose_) std::cout <<"\nfullPath: "<< fullPathHLTFolders[i] << std::endl;
    ig.setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = ig.getSubdirs();
    for (unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {

      if (verbose_) std::cout <<"fullSub: "<<fullSubPathHLTFolders[j] << std::endl;

      if (strcmp(fullSubPathHLTFolders[j].c_str(), "HcalHitsV/SimHitsValidationHcal") == 0) {
	hcalMEs = ig.getContents(fullSubPathHLTFolders[j]);
	if (verbose_) std::cout <<"hltMES size : "<<hcalMEs.size()<<std::endl;
	if( !SimHitsEndjob(hcalMEs) ) std::cout<<"\nError in SimhitEndjob!"<<std::endl<<std::endl;
      }

    }    

  }
  
}

// called after entering the  directory
// hcalMEs are within that directory
int HcalSimHitsClient::SimHitsEndjob(const std::vector<MonitorElement*> &hcalMEs) {
  
  MonitorElement *Occupancy_map[nTime][nType];
  MonitorElement *Energy[nType1];
  MonitorElement *HitEnergyvsieta[nType], *HitTimevsieta[nType];
  std::string divisions[nType]={"HB0","HB1","HE0+z","HE1+z","HE2+z","HE0-z","HE1-z",
				"HE2-z","HO0","HFL0+z","HFS0+z","HFL1+z","HFS1+z",
				"HFL2+z","HFS2+z","HFL3+z","HFS3+z","HFL0-z","HFS0-z",
				"HFL1-z","HFS1-z","HFL2-z","HFS2-z","HFL3-z","HFS3-z"};
  
  std::string time[nTime]={"25","50","100","250"};
  std::string detdivision[nType1]={"HB","HE","HF","HO"};
  char name[40], name1[40], name2[40];

  for (int k=0; k<nType1;k++) {
    Energy[k] = 0;
    for (unsigned int ih=0; ih<hcalMEs.size(); ih++) {
      sprintf (name, "Energy_%s", detdivision[k].c_str());
      if (strcmp(hcalMEs[ih]->getName().c_str(), name) == 0) {
	Energy[k] = hcalMEs[ih];
      }
    }
  }
    
  for (int i=0; i<nTime; i++) {
    for (int j=0; j<nType;j++) {
      Occupancy_map[i][j]= 0;
      for (unsigned int ih=0; ih<hcalMEs.size(); ih++) {
	sprintf (name, "HcalHitE%s%s", time[i].c_str(),divisions[j].c_str());
	if (strcmp(hcalMEs[ih]->getName().c_str(), name) == 0) {
	  Occupancy_map[i][j]= hcalMEs[ih];
	}
      }
    }
  }

  for (int k=0; k<nType;k++) {
    HitEnergyvsieta[k]= 0;
    HitTimevsieta[k]= 0;
    for (unsigned int ih=0; ih<hcalMEs.size(); ih++) {
      sprintf (name1, "HcalHitEta%s",divisions[k].c_str());
      sprintf (name2, "HcalHitTimeAEta%s",divisions[k].c_str());
      if (strcmp(hcalMEs[ih]->getName().c_str(), name1) == 0) {
	HitEnergyvsieta[k]= hcalMEs[ih];
      }
      if (strcmp(hcalMEs[ih]->getName().c_str(), name2) == 0) {
	HitTimevsieta[k]= hcalMEs[ih];
      }
    }
  }
    
  //mean energy 
 
  double nevent = Energy[0]->getEntries();
  if (verbose_) std::cout<<"nevent : "<<nevent<<std::endl;

  float cont[nTime][nType];
  float en[nType1];
  float hitenergy[nType], hittime[nType];
  float fev = float(nevent);
  
  for (int dettype=0; dettype<nType1; dettype++) {
    int nx=Energy[dettype]->getNbinsX();
    for (int i=0; i<=nx; i++) {
      en[dettype]= Energy[dettype]->getBinContent(i)/fev;
      Energy[dettype]->setBinContent(i,en[dettype]);
    }
  }
  
  for (int dettype=0; dettype<nType; dettype++)  {
    if (HitEnergyvsieta[dettype] != 0) {
      int nx1=HitEnergyvsieta[dettype]->getNbinsX();
      for (int i=0; i<=nx1; i++) {
	hitenergy[dettype]= HitEnergyvsieta[dettype]->getBinContent(i)/fev;
	HitEnergyvsieta[dettype]->setBinContent(i,hitenergy[dettype]);
      }
      int nx2= HitTimevsieta[dettype]->getNbinsX();
      for (int i=0; i<=nx2; i++) {
	hittime[dettype]= HitTimevsieta[dettype]->getBinContent(i)/fev;
	HitTimevsieta[dettype]->setBinContent(i,hittime[dettype]);
      }
    }
  }
  
  for (int itime=0; itime<nTime; itime++) {
    for (int det=0; det<nType;det++) {
      if (Occupancy_map[itime][det] != 0) {
	int ny= Occupancy_map[itime][det]->getNbinsY();
	int nx= Occupancy_map[itime][det]->getNbinsX(); 
	for (int i=1; i<nx+1; i++) {
	  for (int j=1; j<ny+1; j++) {
	    cont[itime][det] = Occupancy_map[itime][det]->getBinContent(i,j)/fev;
	    Occupancy_map[itime][det]->setBinContent(i,j,cont[itime][det]);
	  }
	}
      }
    }
  }
 
  return 1;
}

DEFINE_FWK_MODULE(HcalSimHitsClient);
