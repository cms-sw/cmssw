
#include "ShashlikSimHitClient.h"
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


ShashlikSimHitClient::ShashlikSimHitClient(const edm::ParameterSet& iConfig):conf_(iConfig) {

 
  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("ShashlikSimHitClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if (iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if (dbe_) dbe_->setVerbose(0);
  }
  
  verbose_ = true;
  
  outputFile_ = iConfig.getUntrackedParameter<std::string>("OutputFile", "output.root");
  dirName_    = iConfig.getParameter<std::string>("DQMDirName");
  if (dbe_) dbe_->setCurrentFolder(dirName_);
  std::cout<<"Setting of constructor is done"<<std::endl;
}


ShashlikSimHitClient::~ShashlikSimHitClient() { }

void ShashlikSimHitClient::beginJob() { }

void ShashlikSimHitClient::endJob() {
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void ShashlikSimHitClient::beginRun(const edm::Run& run, const edm::EventSetup& c) {
}



void ShashlikSimHitClient::endRun(const edm::Run& , const edm::EventSetup& ) {
  runClient_();
}

//dummy analysis function
void ShashlikSimHitClient::analyze(const edm::Event& , const edm::EventSetup&) { }

void ShashlikSimHitClient::endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup& ) { }

void ShashlikSimHitClient::runClient_() {

  //std::cout<<"about to run client"<<std::endl;
  if (!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);
  
  if (verbose_) std::cout << "\nrunClient" << std::endl; 
  
  std::vector<MonitorElement*> shashlikMEs;
  
  std::vector<std::string> fullPathHLTFolders = dbe_->getSubdirs();
  for (unsigned int i=0;i<fullPathHLTFolders.size();i++) {
    if (verbose_) std::cout <<"\nfullPath: "<< fullPathHLTFolders[i] << std::endl;
    dbe_->setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = dbe_->getSubdirs();
    for (unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {
      
      if (verbose_) std::cout <<"fullSub: "<<fullSubPathHLTFolders[j] << std::endl;
      
      if (strcmp(fullSubPathHLTFolders[j].c_str(), "EcalSimHitsV/Shashlik") == 0) {
	shashlikMEs = dbe_->getContents(fullSubPathHLTFolders[j]);
	if (verbose_) std::cout <<"hltMES size : "<<shashlikMEs.size()<<std::endl;
	
	if( !SimHitsEndjob(shashlikMEs) ) std::cout<<"\nError in SimhitEndjob!"<<std::endl<<std::endl;
      }
      
      else{
	std::cout<<"Directory not found"<<std::endl;
      }

    }    
    
  }
  
}

// called after entering the  directory
// shashlikMEs are within that directory
int ShashlikSimHitClient::SimHitsEndjob(const std::vector<MonitorElement*> &shashlikMEs) {
  
  //std::vector<std::string> divisions = getHistogramTypes();
  //MonitorElement *Occupancy_map[nTime][divisions.size()];
  MonitorElement *Time[nType1], *TimeEwei[nType1], *Energy[nType1], *SumE[nType1];
  MonitorElement  *OccupancymapEwei_time[nType1], *OccupancymapEwei_region[nType1];
  MonitorElement *Occupancymap_region[nType1], *Occupancymap[nType1], *OccupancymapEwei[nType1];
  //MonitorElement *Occupancymap[nType1], *OccupancymapEwei[nType1];
  MonitorElement *OccupancymapEwei_time_region[nTime][nType1];
  

  //std::string time[nTime]={"25","50","100","250"};
  std::string time[nTime]={"25","100"};
  std::string region[2] = {"zM","zP"};
  //std::string detdivision[nType1]={"HB","HE","HF","HO"};
  char name[40]; //, name1[40], name2[40], name3[40], name4[40];
  


  for(int itype=0; itype<nType1; itype++){
    
    Time[itype] = new MonitorElement();
    TimeEwei[itype] = new MonitorElement();
    Energy[itype] = new MonitorElement();
    SumE[itype] = new MonitorElement();
    OccupancymapEwei_time[itype] = new MonitorElement();
    OccupancymapEwei_region[itype] = new MonitorElement();
    Occupancymap_region[itype] = new MonitorElement();
    Occupancymap[itype] = new MonitorElement();
    OccupancymapEwei[itype] = new MonitorElement();
    

  }


    for(int itime=0; itime<nTime; itime++){
      for(int itype=0; itype<nType1; itype++){
	OccupancymapEwei_time_region[itime][itype] = new MonitorElement();
      }
    }


  for (unsigned int ih=0; ih<shashlikMEs.size(); ih++) {

    sprintf (name, "time");
    if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
      Time[0] = shashlikMEs[ih];
    }
    
    sprintf (name, "timeEwei");
    if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
      TimeEwei[0] = shashlikMEs[ih];
    }

    sprintf (name, "energy");
    if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
      Energy[0] = shashlikMEs[ih];
    }

    sprintf (name, "sumE");
    if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
      SumE[0] = shashlikMEs[ih];
    }

  }
    


  
  for (int i=0; i<nTime; i++) {
    
    for (unsigned int ih=0; ih<shashlikMEs.size(); ih++) {
      sprintf (name, "iyVSixEwei_%s", time[i].c_str());
      if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
	OccupancymapEwei_time[i] = shashlikMEs[ih];
      }
    }//for (unsigned int ih=0; ih<shashlikMEs.size(); ih++)

    for( int j=0; j<2; j++){
      for (unsigned int ih=0; ih<shashlikMEs.size(); ih++) {
	sprintf (name, "iyVSixEwei_%s_%s", time[i].c_str(),region[j].c_str());
	if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
	  OccupancymapEwei_time_region[i][j]= shashlikMEs[ih];
	}
      }
    }
  }//for (int i=0; i<nTime; i++)


  for( int j=0; j<2; j++){
    for (unsigned int ih=0; ih<shashlikMEs.size(); ih++) {
      sprintf (name, "iyVSixEwei_%s", region[j].c_str());
	if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
	  OccupancymapEwei_region[j]= shashlikMEs[ih];
	}

	sprintf (name, "iyVSix_%s", region[j].c_str());
	if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
	  Occupancymap_region[j]= shashlikMEs[ih];
	}
    }
  }
  

  for (unsigned int ih=0; ih<shashlikMEs.size(); ih++) {
    sprintf (name, "iyVSix");
    if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
      Occupancymap[0]= shashlikMEs[ih];
    }
    
    sprintf (name, "iyVSixEwei");
    if (strcmp(shashlikMEs[ih]->getName().c_str(), name) == 0) {
      OccupancymapEwei[0]= shashlikMEs[ih];
    }
    
  }
  
    
  


  
  //float cont[nTime][divisions.size()];
  //float en[nType1], tme[nType1];
  //float hitenergy[divisions.size()], hittime[divisions.size()];
  
  
  //energy
  int nx=Energy[0]->getNbinsX();
  double nevent = Energy[0]->getEntries();
  float fev = float(nevent);
  if (verbose_) std::cout<<"nevent : "<<nevent<<std::endl;
  for(int i=0; i<=nx; i++)
    {
      double var= Energy[0]->getBinContent(i)/fev;
      Energy[0]->setBinContent(i,var);
    }
  //time
  nx=Time[0]->getNbinsX();
  nevent = Time[0]->getEntries();
  fev = float(nevent);
  if (verbose_) std::cout<<"nevent : "<<nevent<<std::endl;
  for(int i=0; i<=nx; i++)
    {
      double var=Time[0]->getBinContent(i)/fev;
      Time[0]->setBinContent(i,var);
    }
  //energy weighted time
  nx=TimeEwei[0]->getNbinsX();
  nevent = TimeEwei[0]->getEntries();
  fev = float(nevent);
  if (verbose_) std::cout<<"nevent : "<<nevent<<std::endl;
  for(int i=0; i<=nx; i++)
    {
      double var=TimeEwei[0]->getBinContent(i)/fev;
      TimeEwei[0]->setBinContent(i,var);
    }
  //sumE
  nx=SumE[0]->getNbinsX();
  nevent = SumE[0]->getEntries();
  fev = float(nevent);
  if (verbose_) std::cout<<"nevent : "<<nevent<<std::endl;
  for(int i=0; i<=nx; i++)
    {
      double var= SumE[0]->getBinContent(i)/fev;
      SumE[0]->setBinContent(i,var);
    }
  
  
  for (int itime=0; itime<nTime; itime++) {

    int ny=OccupancymapEwei_time[itime]->getNbinsY(); 
    int nx=OccupancymapEwei_time[itime]->getNbinsX();
    nevent=OccupancymapEwei_time[itime]->getEntries();
    fev=float(nevent);
    for (int i=1; i<=nx; i++) {
      for (int j=1; j<=ny; j++) {
	double var=OccupancymapEwei_time[0]->getBinContent(i,j)/fev;
	OccupancymapEwei_time[0]->setBinContent(i,j,var);
      }//for (int j=1; j<=ny; j++)
    }//for (int i=1; i<=nx; i++)

    for( int idet=0; idet<2; idet++){
      int ny=OccupancymapEwei_time_region[itime][idet]->getNbinsY();
      int nx=OccupancymapEwei_time_region[itime][idet]->getNbinsX();
      nevent=OccupancymapEwei_time_region[itime][idet]->getEntries();
      fev = float(nevent);
      for (int i=1; i<=nx; i++) {
	for (int j=1; j<=ny; j++) {
	  double var = OccupancymapEwei_time_region[itime][idet]->getBinContent(i,j)/fev ;
	  OccupancymapEwei_time_region[itime][idet]->setBinContent(i,j,var);
	}
      }
    }
    //std::cout<<"Out of the detector loop-----everything is filled"<<std::endl;
	
  }//for (int itime=0; itime<nTime; itime++)
  
  
  for( int idet=0; idet<2; idet++){
    
    int ny=OccupancymapEwei_region[idet]->getNbinsY();
    int nx=OccupancymapEwei_region[idet]->getNbinsX();
    nevent=OccupancymapEwei_region[idet]->getEntries();
    fev=float(nevent);
    for (int i=1; i<=nx; i++) {
      for (int j=1; j<=ny; j++) {
	double var=OccupancymapEwei_region[idet]->getBinContent(i,j)/fev;
	OccupancymapEwei_region[idet]->setBinContent(i,j,var);
      }//for (int j=1; j<=ny; j++)
    }//for (int i=1; i<=nx; i++) 

    ny=Occupancymap_region[idet]->getNbinsY();
    nx=Occupancymap_region[idet]->getNbinsX();
    nevent=Occupancymap_region[idet]->getEntries();
    fev=float(nevent);
    for (int i=1; i<=nx; i++) {
      for (int j=1; j<=ny; j++) {
	double var=Occupancymap_region[idet]->getBinContent(i,j)/fev;
	Occupancymap_region[idet]->setBinContent(i,j,var);
      }//for (int j=1; j<=ny; j++)
    }//for (int i=1; i<=nx; i++)
  }//for( int idet=0; idet<2; j++)

  
  int ny=Occupancymap[0]->getNbinsY();
  nx=Occupancymap[0]->getNbinsX();
  nevent=Occupancymap[0]->getEntries();
  fev=float(nevent);
  for (int i=1; i<=nx; i++) {
    for (int j=1; j<=ny; j++) {
      double var=Occupancymap[0]->getBinContent(i,j)/fev;
      Occupancymap[0]->setBinContent(i,j,var);
    }//for (int j=1; j<=ny; j++)
  }//for (int i=1; i<=nx; i++)
  
  ny=OccupancymapEwei[0]->getNbinsY();
  nx=OccupancymapEwei[0]->getNbinsX();
  nevent=OccupancymapEwei[0]->getEntries();
  fev=float(nevent);
  for (int i=1; i<=nx; i++) {
    for (int j=1; j<=ny; j++) {
      double var=OccupancymapEwei[0]->getBinContent(i,j)/fev;
      OccupancymapEwei[0]->setBinContent(i,j,var);
    }//for (int j=1; j<=ny; j++) 
  }//for (int i=1; i<=nx; i++) 
  
    
  
  return 1;
}




DEFINE_FWK_MODULE(ShashlikSimHitClient);
