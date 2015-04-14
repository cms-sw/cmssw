#include "Validation/HcalRecHits/interface/NoiseRatesClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"

NoiseRatesClient::NoiseRatesClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{

  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  debug_ = false;
  verbose_ = false;

  dirName_=iConfig.getParameter<std::string>("DQMDirName");
 
}


NoiseRatesClient::~NoiseRatesClient()
{ 
  
}



void NoiseRatesClient::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig)
{
  runClient_(ib,ig);
}

void NoiseRatesClient::runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig)
{
  ig.setCurrentFolder(dirName_);

  if (verbose_) std::cout << "\nrunClient" << std::endl; 

  std::vector<MonitorElement*> hcalMEs;

  // Since out folders are fixed to three, we can just go over these three folders
  // i.e., CaloTowersV/CaloTowersTask, HcalRecHitsV/HcalRecHitTask, NoiseRatesV/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = ig.getSubdirs();
  for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {

    if (verbose_) std::cout <<"\nfullPath: "<< fullPathHLTFolders[i] << std::endl;
    ig.setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = ig.getSubdirs();
    for(unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {

      if (verbose_) std::cout <<"fullSub: "<<fullSubPathHLTFolders[j] << std::endl;

      if( strcmp(fullSubPathHLTFolders[j].c_str(), "NoiseRatesV/NoiseRatesTask") ==0  ){
         hcalMEs = ig.getContents(fullSubPathHLTFolders[j]);
         if (verbose_) std::cout <<"hltMES size : "<<hcalMEs.size()<<std::endl;
         if( !NoiseRatesEndjob(hcalMEs) ) std::cout<<"\nError in NoiseRatesEndjob!"<<std::endl<<std::endl;
      }

    }    

  }

}

// called after entering the NoiseRatesV/NoiseRatesTask directory
// hcalMEs are within that directory
int NoiseRatesClient::NoiseRatesEndjob(const std::vector<MonitorElement*> &hcalMEs){

   int useAllHistos = 0;
   MonitorElement* hLumiBlockCount =0;
   for(unsigned int ih=0; ih<hcalMEs.size(); ih++){
      if( strcmp(hcalMEs[ih]->getName().c_str(), "hLumiBlockCount") ==0  ){
         hLumiBlockCount = hcalMEs[ih];
         useAllHistos =1;
      } 
   } 
   if( useAllHistos !=0 && useAllHistos !=1 ) return 0;

// FIXME: dummy lumiCountMap.size since hLumiBlockCount is disabled
// in a general case.
   int lumiCountMapsize = -1; // dummy
   if (useAllHistos) hLumiBlockCount->Fill(0.0, lumiCountMapsize);

   return 1;

}

DEFINE_FWK_MODULE(NoiseRatesClient);
