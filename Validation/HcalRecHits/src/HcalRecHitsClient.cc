#include "Validation/HcalRecHits/interface/HcalRecHitsClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

HcalRecHitsClient::HcalRecHitsClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{

  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");

  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("HcalRecHitsClient") << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", true)) {
    if(dbe_) dbe_->setVerbose(0);
  }
 
  debug_ = false;
  verbose_ = false;

  // false for regular relval and true for SLHC relval
  doSLHC_ = iConfig.getUntrackedParameter<bool>("doSLHC", false);


  dirName_=iConfig.getParameter<std::string>("DQMDirName");
  if(dbe_) dbe_->setCurrentFolder(dirName_);
 
}


HcalRecHitsClient::~HcalRecHitsClient()
{ 
  
}

void HcalRecHitsClient::beginJob()
{
 

}

void HcalRecHitsClient::endJob() 
{
   if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void HcalRecHitsClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
 
}


void HcalRecHitsClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  runClient_();
}

//dummy analysis function
void HcalRecHitsClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
}

void HcalRecHitsClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 
//  runClient_();
}

void HcalRecHitsClient::runClient_()
{
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything
  dbe_->setCurrentFolder(dirName_);

  if (verbose_) std::cout << "\nrunClient" << std::endl; 

  std::vector<MonitorElement*> hcalMEs;

  // Since out folders are fixed to three, we can just go over these three folders
  // i.e., CaloTowersV/CaloTowersTask, HcalRecHitsV/HcalRecHitTask, NoiseRatesV/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {

    if (verbose_) std::cout <<"\nfullPath: "<< fullPathHLTFolders[i] << std::endl;
    dbe_->setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = dbe_->getSubdirs();
    for(unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {

      if (verbose_) std::cout <<"fullSub: "<<fullSubPathHLTFolders[j] << std::endl;

      if( strcmp(fullSubPathHLTFolders[j].c_str(), "HcalRecHitsV/HcalRecHitTask") ==0  ){
         hcalMEs = dbe_->getContents(fullSubPathHLTFolders[j]);
         if (verbose_) std::cout <<"hltMES size : "<<hcalMEs.size()<<std::endl;
         if( !HcalRecHitsEndjob(hcalMEs) ) std::cout<<"\nError in HcalRecHitsEndjob!"<<std::endl<<std::endl;
      }

    }    

  }

}


// called after entering the HcalRecHitsV/HcalRecHitTask directory
// hcalMEs are within that directory
int HcalRecHitsClient::HcalRecHitsEndjob(const std::vector<MonitorElement*> &hcalMEs){

   int useAllHistos = 0, subdet =5;

// for ZS ...
//   MonitorElement* emap_min_ME =0;
   MonitorElement* ZS_HO=0, *ZS_seqHO=0;
   MonitorElement* ZS_HB1=0, *ZS_seqHB1=0, *ZS_HB2=0, *ZS_seqHB2=0; 
   MonitorElement* ZS_HF1=0, *ZS_seqHF1=0, *ZS_HF2=0, *ZS_seqHF2=0; 
   MonitorElement* ZS_HE1=0, *ZS_seqHE1=0, *ZS_HE2=0, *ZS_seqHE2=0, *ZS_HE3=0, *ZS_seqHE3=0;
   MonitorElement* map_depth1 =0, *map_depth2 =0, *map_depth3 =0, *map_depth4 =0, *map_depth5 =0, *map_depth6 =0, *map_depth7 =0;
// others 
   MonitorElement* Nhf=0;
   MonitorElement* emap_depth1 =0, *emap_depth2 =0, *emap_depth3 =0, *emap_depth4 =0, *emap_depth5 =0, *emap_depth6 =0, *emap_depth7 =0; 
   MonitorElement* occupancy_seqHB1 =0, *occupancy_seqHB2 =0, *occupancy_seqHB3 =0, *occupancy_seqHB4 =0, *occupancy_seqHB5 =0, *occupancy_seqHB6 =0, *occupancy_seqHB7 =0; 
   MonitorElement* occupancy_seqHE1 =0, *occupancy_seqHE2 =0, *occupancy_seqHE3 =0, *occupancy_seqHE4 =0, *occupancy_seqHE5 =0, *occupancy_seqHE6 =0, *occupancy_seqHE7 =0;
   MonitorElement* occupancy_seqHF1 =0, *occupancy_seqHF2 =0; 
   MonitorElement* occupancy_seqHO =0;
   MonitorElement* emean_seqHB1 =0, *emean_seqHB2 =0, *emean_seqHB3 =0, *emean_seqHB4 =0, *emean_seqHB5 =0, *emean_seqHB6 =0, *emean_seqHB7 =0; 
   MonitorElement* emean_seqHE1 =0, *emean_seqHE2 =0, *emean_seqHE3 =0, *emean_seqHE4 =0, *emean_seqHE5 =0, *emean_seqHE6 =0, *emean_seqHE7 =0;
   MonitorElement* emean_seqHF1 =0, *emean_seqHF2 =0; 
   MonitorElement* emean_seqHO =0;

   MonitorElement* RMS_seq_HB1 =0, *RMS_seq_HB2 =0, *RMS_seq_HB3 =0, *RMS_seq_HB4 =0, *RMS_seq_HB5 =0, *RMS_seq_HB6 =0, *RMS_seq_HB7 =0; 
   MonitorElement* RMS_seq_HE1 =0, *RMS_seq_HE2 =0, *RMS_seq_HE3 =0, *RMS_seq_HE4 =0, *RMS_seq_HE5 =0, *RMS_seq_HE6 =0, *RMS_seq_HE7 =0;
   MonitorElement* RMS_seq_HF1 =0, *RMS_seq_HF2 =0; 
   MonitorElement* RMS_seq_HO =0;

   MonitorElement *occupancy_map_HO =0;
   MonitorElement* occupancy_map_HB1 =0, *occupancy_map_HB2 =0, *occupancy_map_HB3 =0, *occupancy_map_HB4 =0, *occupancy_map_HB5 =0, *occupancy_map_HB6 =0, *occupancy_map_HB7 =0;
   MonitorElement* occupancy_map_HF1 =0, *occupancy_map_HF2 =0;
   MonitorElement* occupancy_map_HE1 =0, *occupancy_map_HE2 =0, *occupancy_map_HE3 =0, *occupancy_map_HE4 =0, *occupancy_map_HE5 =0, *occupancy_map_HE6 =0, *occupancy_map_HE7 =0;

   MonitorElement* emean_vs_ieta_HB1 =0, *emean_vs_ieta_HB2 =0, *emean_vs_ieta_HB3 =0, *emean_vs_ieta_HB4 =0, *emean_vs_ieta_HB5 =0, *emean_vs_ieta_HB6 =0, *emean_vs_ieta_HB7 =0; 
   MonitorElement* emean_vs_ieta_HE1 =0, *emean_vs_ieta_HE2 =0, *emean_vs_ieta_HE3 =0, *emean_vs_ieta_HE4 =0, *emean_vs_ieta_HE5 =0, *emean_vs_ieta_HE6 =0, *emean_vs_ieta_HE7 =0;
   //MonitorElement* emean_vs_ieta_HF1 =0, *emean_vs_ieta_HF2 =0; 
   //MonitorElement* emean_vs_ieta_HO =0;
   MonitorElement* RMS_vs_ieta_HB1 =0, *RMS_vs_ieta_HB2 =0, *RMS_vs_ieta_HB3 =0, *RMS_vs_ieta_HB4 =0, *RMS_vs_ieta_HB5 =0, *RMS_vs_ieta_HB6 =0, *RMS_vs_ieta_HB7 =0; 
   MonitorElement* RMS_vs_ieta_HE1 =0, *RMS_vs_ieta_HE2 =0, *RMS_vs_ieta_HE3 =0, *RMS_vs_ieta_HE4 =0, *RMS_vs_ieta_HE5 =0, *RMS_vs_ieta_HE6 =0, *RMS_vs_ieta_HE7 =0;
   MonitorElement* RMS_vs_ieta_HF1 =0, *RMS_vs_ieta_HF2 =0; 
   MonitorElement* RMS_vs_ieta_HO =0;
   MonitorElement* occupancy_vs_ieta_HB1 =0, *occupancy_vs_ieta_HB2 =0, *occupancy_vs_ieta_HB3 =0, *occupancy_vs_ieta_HB4 =0, *occupancy_vs_ieta_HB5 =0, *occupancy_vs_ieta_HB6 =0, *occupancy_vs_ieta_HB7 =0; 
   MonitorElement* occupancy_vs_ieta_HE1 =0, *occupancy_vs_ieta_HE2 =0, *occupancy_vs_ieta_HE3 =0, *occupancy_vs_ieta_HE4 =0, *occupancy_vs_ieta_HE5 =0, *occupancy_vs_ieta_HE6 =0, *occupancy_vs_ieta_HE7 =0;
   MonitorElement* occupancy_vs_ieta_HF1 =0, *occupancy_vs_ieta_HF2 =0; 
   MonitorElement* occupancy_vs_ieta_HO =0;

   MonitorElement* RecHit_StatusWord_HB =0, *RecHit_StatusWord_HE=0, *RecHit_StatusWord_HO =0, *RecHit_StatusWord_HF =0, *RecHit_StatusWord_HF67 =0;
   MonitorElement* RecHit_Aux_StatusWord_HB =0, *RecHit_Aux_StatusWord_HE=0, *RecHit_Aux_StatusWord_HO =0, *RecHit_Aux_StatusWord_HF =0;

   for(unsigned int ih=0; ih<hcalMEs.size(); ih++){
      if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_map_depth1") ==0  ){
         useAllHistos =1; subdet =6;
      }
      if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB1") ==0  ){
         useAllHistos =1;
      }
//      if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_min_ME") ==0  ){ emap_min_ME = hcalMEs[ih]; }
      if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HO") ==0  ){ ZS_HO = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HO") ==0  ){ ZS_seqHO = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HB1") ==0  ){ ZS_HB1 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HB1") ==0  ){ ZS_seqHB1 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HB2") ==0  ){ ZS_HB2 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HB2") ==0  ){ ZS_seqHB2 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HF1") ==0  ){ ZS_HF1 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HF1") ==0  ){ ZS_seqHF1 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HF2") ==0  ){ ZS_HF2 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HF2") ==0  ){ ZS_seqHF2 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HE1") ==0  ){ ZS_HE1 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HE1") ==0  ){ ZS_seqHE1 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HE2") ==0  ){ ZS_HE2 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HE2") ==0  ){ ZS_seqHE2 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_simple1D_HE3") ==0  ){ ZS_HE3 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_sequential1D_HE3") ==0  ){ ZS_seqHE3 = hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_map_depth1") ==0  ){ map_depth1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_map_depth2") ==0  ){ map_depth2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_map_depth3") ==0  ){ map_depth3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "ZSmin_map_depth4") ==0  ){ map_depth4= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "N_HF") ==0  ){ Nhf= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_depth1") ==0  ){ emap_depth1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_depth2") ==0  ){ emap_depth2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_depth3") ==0  ){ emap_depth3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_depth4") ==0  ){ emap_depth4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_depth5") ==0  ){ emap_depth5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_depth6") ==0  ){ emap_depth6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emap_depth7") ==0  ){ emap_depth7= hcalMEs[ih]; }


      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HB1") ==0  ){ occupancy_seqHB1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HB2") ==0  ){ occupancy_seqHB2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HB3") ==0  ){ occupancy_seqHB3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HB4") ==0  ){ occupancy_seqHB4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HB5") ==0  ){ occupancy_seqHB5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HB6") ==0  ){ occupancy_seqHB6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HB7") ==0  ){ occupancy_seqHB7= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HE1") ==0  ){ occupancy_seqHE1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HE2") ==0  ){ occupancy_seqHE2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HE3") ==0  ){ occupancy_seqHE3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HE4") ==0  ){ occupancy_seqHE4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HE5") ==0  ){ occupancy_seqHE5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HE6") ==0  ){ occupancy_seqHE6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HE7") ==0  ){ occupancy_seqHE7= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HF1") ==0  ){ occupancy_seqHF1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HF2") ==0  ){ occupancy_seqHF2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occ_sequential1D_HO") ==0  ){ occupancy_seqHO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HB1") ==0  ){ emean_seqHB1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HB2") ==0  ){ emean_seqHB2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HB3") ==0  ){ emean_seqHB3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HB4") ==0  ){ emean_seqHB4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HB5") ==0  ){ emean_seqHB5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HB6") ==0  ){ emean_seqHB6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HB7") ==0  ){ emean_seqHB7= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HE1") ==0  ){ emean_seqHE1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HE2") ==0  ){ emean_seqHE2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HE3") ==0  ){ emean_seqHE3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HE4") ==0  ){ emean_seqHE4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HE5") ==0  ){ emean_seqHE5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HE6") ==0  ){ emean_seqHE6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HE7") ==0  ){ emean_seqHE7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HF1") ==0  ){ emean_seqHF1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HF2") ==0  ){ emean_seqHF2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_seq_HO") ==0  ){ emean_seqHO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HB1") ==0  ){ RMS_seq_HB1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HB2") ==0  ){ RMS_seq_HB2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HB3") ==0  ){ RMS_seq_HB3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HB4") ==0  ){ RMS_seq_HB4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HB5") ==0  ){ RMS_seq_HB5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HB6") ==0  ){ RMS_seq_HB6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HB7") ==0  ){ RMS_seq_HB7= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HE1") ==0  ){ RMS_seq_HE1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HE2") ==0  ){ RMS_seq_HE2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HE3") ==0  ){ RMS_seq_HE3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HE4") ==0  ){ RMS_seq_HE4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HE5") ==0  ){ RMS_seq_HE5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HE6") ==0  ){ RMS_seq_HE6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HE7") ==0  ){ RMS_seq_HE7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HF1") ==0  ){ RMS_seq_HF1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HF2") ==0  ){ RMS_seq_HF2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_seq_HO") ==0  ){ RMS_seq_HO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HB1") ==0  ){ occupancy_map_HB1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HB2") ==0  ){ occupancy_map_HB2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HB3") ==0  ){ occupancy_map_HB3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HB4") ==0  ){ occupancy_map_HB4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HB5") ==0  ){ occupancy_map_HB5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HB6") ==0  ){ occupancy_map_HB6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HB7") ==0  ){ occupancy_map_HB7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HE1") ==0  ){ occupancy_map_HE1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HE2") ==0  ){ occupancy_map_HE2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HE3") ==0  ){ occupancy_map_HE3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HE4") ==0  ){ occupancy_map_HE4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HE5") ==0  ){ occupancy_map_HE5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HE6") ==0  ){ occupancy_map_HE6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HE7") ==0  ){ occupancy_map_HE7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HF1") ==0  ){ occupancy_map_HF1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HF2") ==0  ){ occupancy_map_HF2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_map_HO") ==0  ){ occupancy_map_HO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HB1") ==0  ){ emean_vs_ieta_HB1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HB2") ==0  ){ emean_vs_ieta_HB2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HB3") ==0  ){ emean_vs_ieta_HB3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HB4") ==0  ){ emean_vs_ieta_HB4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HB5") ==0  ){ emean_vs_ieta_HB5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HB6") ==0  ){ emean_vs_ieta_HB6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HB7") ==0  ){ emean_vs_ieta_HB7= hcalMEs[ih]; }      

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HE1") ==0  ){ emean_vs_ieta_HE1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HE2") ==0  ){ emean_vs_ieta_HE2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HE3") ==0  ){ emean_vs_ieta_HE3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HE4") ==0  ){ emean_vs_ieta_HE4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HE5") ==0  ){ emean_vs_ieta_HE5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HE6") ==0  ){ emean_vs_ieta_HE6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HE7") ==0  ){ emean_vs_ieta_HE7= hcalMEs[ih]; }     

      //else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HF1") ==0  ){ emean_vs_ieta_HF1= hcalMEs[ih]; }
      //else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HF2") ==0  ){ emean_vs_ieta_HF2= hcalMEs[ih]; }
      //else if( strcmp(hcalMEs[ih]->getName().c_str(), "emean_vs_ieta_HO") ==0  ){ emean_vs_ieta_HO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB1") ==0  ){ RMS_vs_ieta_HB1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB2") ==0  ){ RMS_vs_ieta_HB2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB3") ==0  ){ RMS_vs_ieta_HB3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB4") ==0  ){ RMS_vs_ieta_HB4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB5") ==0  ){ RMS_vs_ieta_HB5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB6") ==0  ){ RMS_vs_ieta_HB6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HB7") ==0  ){ RMS_vs_ieta_HB7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HE1") ==0  ){ RMS_vs_ieta_HE1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HE2") ==0  ){ RMS_vs_ieta_HE2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HE3") ==0  ){ RMS_vs_ieta_HE3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HE4") ==0  ){ RMS_vs_ieta_HE4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HE5") ==0  ){ RMS_vs_ieta_HE5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HE6") ==0  ){ RMS_vs_ieta_HE6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HE7") ==0  ){ RMS_vs_ieta_HE7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HF1") ==0  ){ RMS_vs_ieta_HF1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HF2") ==0  ){ RMS_vs_ieta_HF2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "RMS_vs_ieta_HO") ==0  ){ RMS_vs_ieta_HO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HB1") ==0  ){ occupancy_vs_ieta_HB1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HB2") ==0  ){ occupancy_vs_ieta_HB2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HB3") ==0  ){ occupancy_vs_ieta_HB3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HB4") ==0  ){ occupancy_vs_ieta_HB4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HB5") ==0  ){ occupancy_vs_ieta_HB5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HB6") ==0  ){ occupancy_vs_ieta_HB6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HB7") ==0  ){ occupancy_vs_ieta_HB7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HE1") ==0  ){ occupancy_vs_ieta_HE1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HE2") ==0  ){ occupancy_vs_ieta_HE2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HE3") ==0  ){ occupancy_vs_ieta_HE3= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HE4") ==0  ){ occupancy_vs_ieta_HE4= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HE5") ==0  ){ occupancy_vs_ieta_HE5= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HE6") ==0  ){ occupancy_vs_ieta_HE6= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HE7") ==0  ){ occupancy_vs_ieta_HE7= hcalMEs[ih]; }

      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HF1") ==0  ){ occupancy_vs_ieta_HF1= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HF2") ==0  ){ occupancy_vs_ieta_HF2= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "occupancy_vs_ieta_HO") ==0  ){ occupancy_vs_ieta_HO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_StatusWord_HB") ==0  ){ RecHit_StatusWord_HB= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_StatusWord_HE") ==0  ){ RecHit_StatusWord_HE= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_StatusWord_HO") ==0  ){ RecHit_StatusWord_HO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_StatusWord_HF") ==0  ){ RecHit_StatusWord_HF= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_StatusWord_HF67") ==0  ){ RecHit_StatusWord_HF67= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_Aux_StatusWord_HB") ==0  ){ RecHit_Aux_StatusWord_HB= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_Aux_StatusWord_HE") ==0  ){ RecHit_Aux_StatusWord_HE= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_Aux_StatusWord_HO") ==0  ){ RecHit_Aux_StatusWord_HO= hcalMEs[ih]; }
      else if( strcmp(hcalMEs[ih]->getName().c_str(), "HcalRecHitTask_RecHit_Aux_StatusWord_HF") ==0  ){ RecHit_Aux_StatusWord_HF= hcalMEs[ih]; }
   } 
   if( useAllHistos !=0 && useAllHistos !=1 ) return 0;

   //None of the ZS stuff necessary for drawn histograms
   if (subdet==6 && useAllHistos) {
// FIXME: a dummy emap_min! No solutions yet found.
// Since useAllHistos is set to disable it in our ususal validation, it's left here to be fixed later...
      double emap_min[82][72][4][4];
/* NOT a valid solution 
      for(unsigned int i1=0; i1 <82; i1++){
         for(unsigned int i2=0; i2<72; i2++){
            for(unsigned int i3=0; i3<4; i3++){
               for(unsigned int i4=0; i4<4; i4++){
                  int idx = i1 + i2*82 + i3*(82*72) + i4*(82*72*4);
                  emap_min[i1][i2][i3][i4] = emap_min_ME->GetBinContent(idx+1);     
               }
            }
         }
       }
*/
       for (unsigned int i1 = 0;  i1 < 82; i1++) {
          for (unsigned int i2 = 0;  i2 < 72; i2++) {

             int index = (i1-41) * 72 + i2;
	
             double e = emap_min[i1][i2][0][0];
             if( e < 10000.) {
	        ZS_HB1->Fill(e);
	        ZS_seqHB1->Fill(double(index),e);
	     }
	     e = emap_min[i1][i2][1][0];
	     if( e < 10000.) {
	        ZS_HB2->Fill(e);
	        ZS_seqHB2->Fill(double(index),e);
	     }
	
	     e = emap_min[i1][i2][0][1];
	     if( e < 10000.) {
	        ZS_HE1->Fill(e);
	        ZS_seqHE1->Fill(double(index),e);
	     }
	     e = emap_min[i1][i2][1][1];
	     if( e < 10000.) {
	        ZS_HE2->Fill(e);
	        ZS_seqHE2->Fill(double(index),e);
	     }
	     e = emap_min[i1][i2][2][1];
	     if( e < 10000.) {
	        ZS_HE3->Fill(e);
	        ZS_seqHE3->Fill(double(index),e);
	     }
	
	     e = emap_min[i1][i2][3][2];
	     if( e < 10000.) {
	        ZS_HO->Fill(e);
	        ZS_seqHO->Fill(double(index),e);
	     }
	
	     e = emap_min[i1][i2][0][3];
	     if( e < 10000.) {
	        ZS_HF1->Fill(e);
	        ZS_seqHF1->Fill(double(index),e);
	     }
	
	     e = emap_min[i1][i2][1][3];
	     if( e < 10000.) {
	        ZS_HF2->Fill(e);
	        ZS_seqHF2->Fill(double(index),e);
	     }

          
            unsigned int n_depth = 4;
            if (doSLHC_){n_depth = 7;}
	
	     for (unsigned int i3 = 0;  i3 < n_depth;  i3++) {  // depth
	        double emin = 100000.;
	        for (unsigned int i4 = 0;  i4 < 4;  i4++) {  // subdet
	           /*
	             std::cout << "* ieta, iphi, depth, sub = " 
		         << i1 << ", " << i2 << ", " << i3 << ", " << i4
		         << "  emap_min = " << emap_min [i1][i2][i3][i4]
		         << std::endl;
	           */
	           if ( emin > emap_min [i1][i2][i3][i4]) 
	           emin = emap_min [i1][i2][i3][i4];
	        }

                int ieta = i1-41;
	        if( i3 == 0 && emin < 10000.) {
	           map_depth1->Fill(double(ieta),double(i2),emin);
	           /*
	             std::cout << "* Fill map_depth1 " << double(ieta) << " "  
		        << double(i2) << "  with " << emin <<  std::endl;
	           */
	        }
	        if( i3 == 1 && emin < 10000.)
	           map_depth2->Fill(double(ieta),double(i2),emin);
	        if( i3 == 2 && emin < 10000.) 
	           map_depth3->Fill(double(ieta),double(i2),emin);
                if( i3 == 3 && emin < 10000.) 
	           map_depth4->Fill(double(ieta),double(i2),emin);
               if (doSLHC_){ 
                  if( i3 == 4 && emin < 10000.)
                     map_depth5->Fill(double(ieta),double(i2),emin);
                  if( i3 == 5 && emin < 10000.)
                     map_depth6->Fill(double(ieta),double(i2),emin);
                  if( i3 == 6 && emin < 10000.)
                    map_depth7->Fill(double(ieta),double(i2),emin);
                }

	    }
         }
      } 
   }
  // mean energies and occupancies evaluation
   else {

      double nevtot = Nhf->getEntries();
      if(verbose_) std::cout<<"nevtot : "<<nevtot<<std::endl;

      int nx = occupancy_map_HB1->getNbinsX();
      int ny = occupancy_map_HB1->getNbinsY();

      float cnorm;
      float fev = float (nevtot);
      //    std::cout << "*** nevtot " <<  nevtot << std::endl; 

      float sumphi_hb1, sumphi_hb2, sumphi_hb3, sumphi_hb4, sumphi_hb5, sumphi_hb6, sumphi_hb7, sumphi_he1, sumphi_he2, sumphi_he3, sumphi_he4, sumphi_he5, sumphi_he6, sumphi_he7,
            sumphi_ho, sumphi_hf1, sumphi_hf2;
      /*
      if(nx != 82 || ny != 72) 
            std::cout << "*** problem with binning " << std::endl;
      */
      float phi_factor;  

      // First - special <E> maps
      int nx1 = emap_depth1->getNbinsX();    
      int ny1 = emap_depth1->getNbinsY();
      for (int i = 1; i <= nx1; i++) {      
	for (int j = 1; j <= ny1; j++) {      
	    cnorm = emap_depth1->getBinContent(i,j) / fev;
            emap_depth1->setBinContent(i,j,cnorm);
	    cnorm = emap_depth2->getBinContent(i,j) / fev;
            emap_depth2->setBinContent(i,j,cnorm);
	    cnorm = emap_depth3->getBinContent(i,j) / fev;
            emap_depth3->setBinContent(i,j,cnorm);
	    cnorm = emap_depth4->getBinContent(i,j) / fev;
            emap_depth4->setBinContent(i,j,cnorm);
 
            if (doSLHC_){
               cnorm = emap_depth5->getBinContent(i,j) / fev;
               emap_depth5->setBinContent(i,j,cnorm);
               cnorm = emap_depth6->getBinContent(i,j) / fev;
               emap_depth6->setBinContent(i,j,cnorm);
               cnorm = emap_depth7->getBinContent(i,j) / fev;
               emap_depth7->setBinContent(i,j,cnorm);
         }
	}
      }

      // Second: all others regular maps
      for (int i = 1; i <= nx; i++) {
         sumphi_hb1 = 0.;
         sumphi_hb2 = 0.;
         sumphi_hb3 = 0.;
         sumphi_hb4 = 0.;
         sumphi_hb5 = 0.;
         sumphi_hb6 = 0.;
         sumphi_hb7 = 0.;
         sumphi_he1 = 0.;
         sumphi_he2 = 0.;
         sumphi_he3 = 0.;
         sumphi_he4 = 0.;
         sumphi_he5 = 0.;
         sumphi_he6 = 0.;
         sumphi_he7 = 0.;
         sumphi_ho  = 0.; 
         sumphi_hf1 = 0.;
         sumphi_hf2 = 0.;
      
         for (int j = 1; j <= ny; j++) {
	
            int index = (i-42) * ny + j-1;

	    //Occupancies (needed for occ vs ieta histos)
	    cnorm = occupancy_map_HB1->getBinContent(i,j) / fev;   
	    occupancy_map_HB1->setBinContent(i,j,cnorm);
	
	    cnorm = occupancy_map_HB2->getBinContent(i,j) / fev;   
	    occupancy_map_HB2->setBinContent(i,j,cnorm);

            if (doSLHC_){
               cnorm = occupancy_map_HB3->getBinContent(i,j) / fev;
               occupancy_map_HB3->setBinContent(i,j,cnorm);
               cnorm = occupancy_map_HB4->getBinContent(i,j) / fev;
               occupancy_map_HB4->setBinContent(i,j,cnorm);
               cnorm = occupancy_map_HB5->getBinContent(i,j) / fev;
               occupancy_map_HB5->setBinContent(i,j,cnorm);
               cnorm = occupancy_map_HB6->getBinContent(i,j) / fev;
               occupancy_map_HB6->setBinContent(i,j,cnorm);
               cnorm = occupancy_map_HB7->getBinContent(i,j) / fev;
               occupancy_map_HB7->setBinContent(i,j,cnorm);
            }
	
	    cnorm = occupancy_map_HE1->getBinContent(i,j) / fev;   
	    occupancy_map_HE1->setBinContent(i,j,cnorm);
	
	    cnorm = occupancy_map_HE2->getBinContent(i,j) / fev;   
	    occupancy_map_HE2->setBinContent(i,j,cnorm);

            cnorm = occupancy_map_HE3->getBinContent(i,j) / fev;
            occupancy_map_HE3->setBinContent(i,j,cnorm);
	
            if (doSLHC_){
	       cnorm = occupancy_map_HE4->getBinContent(i,j) / fev;   
	       occupancy_map_HE4->setBinContent(i,j,cnorm);
               cnorm = occupancy_map_HE5->getBinContent(i,j) / fev;
               occupancy_map_HE5->setBinContent(i,j,cnorm);
               cnorm = occupancy_map_HE6->getBinContent(i,j) / fev;
               occupancy_map_HE6->setBinContent(i,j,cnorm);
               cnorm = occupancy_map_HE7->getBinContent(i,j) / fev;
               occupancy_map_HE7->setBinContent(i,j,cnorm);
            }

	    cnorm = occupancy_map_HO->getBinContent(i,j) / fev;   
	    occupancy_map_HO->setBinContent(i,j,cnorm);
	
	    cnorm = occupancy_map_HF1->getBinContent(i,j) / fev;   
	    occupancy_map_HF1->setBinContent(i,j,cnorm);
	
	    cnorm = occupancy_map_HF2->getBinContent(i,j) / fev;   
	    occupancy_map_HF2->setBinContent(i,j,cnorm);

	    sumphi_hb1 += occupancy_map_HB1->getBinContent(i,j);
	    sumphi_hb2 += occupancy_map_HB2->getBinContent(i,j);
           
            if (doSLHC_){
               sumphi_hb3 += occupancy_map_HB3->getBinContent(i,j);
               sumphi_hb4 += occupancy_map_HB4->getBinContent(i,j);
               sumphi_hb5 += occupancy_map_HB5->getBinContent(i,j);
               sumphi_hb6 += occupancy_map_HB6->getBinContent(i,j);
               sumphi_hb7 += occupancy_map_HB7->getBinContent(i,j);
            }

	    sumphi_he1 += occupancy_map_HE1->getBinContent(i,j);
	    sumphi_he2 += occupancy_map_HE2->getBinContent(i,j);
	    sumphi_he3 += occupancy_map_HE3->getBinContent(i,j);
      
            if (doSLHC_){
               sumphi_he4 += occupancy_map_HE4->getBinContent(i,j);
               sumphi_he5 += occupancy_map_HE5->getBinContent(i,j);
               sumphi_he6 += occupancy_map_HE6->getBinContent(i,j);
               sumphi_he7 += occupancy_map_HE7->getBinContent(i,j);
            }

	    sumphi_ho  += occupancy_map_HO->getBinContent(i,j);
	    sumphi_hf1 += occupancy_map_HF1->getBinContent(i,j);
	    sumphi_hf2 += occupancy_map_HF2->getBinContent(i,j);
	
	    // Occupancies - not in main drawn set of histos
            if(useAllHistos){
              occupancy_seqHB1->Fill(double(index),cnorm);
	      occupancy_seqHB2->Fill(double(index),cnorm);
              occupancy_seqHB3->Fill(double(index),cnorm);
              occupancy_seqHB4->Fill(double(index),cnorm);
              occupancy_seqHB5->Fill(double(index),cnorm);
              occupancy_seqHB6->Fill(double(index),cnorm);
              occupancy_seqHB7->Fill(double(index),cnorm);
     
	      occupancy_seqHE1->Fill(double(index),cnorm);
	      occupancy_seqHE2->Fill(double(index),cnorm);
	      occupancy_seqHE3->Fill(double(index),cnorm);
              occupancy_seqHE4->Fill(double(index),cnorm);
              occupancy_seqHE5->Fill(double(index),cnorm);
              occupancy_seqHE6->Fill(double(index),cnorm);
              occupancy_seqHE7->Fill(double(index),cnorm);

	      occupancy_seqHO->Fill(double(index),cnorm);
	      occupancy_seqHF1->Fill(double(index),cnorm);
	      occupancy_seqHF2->Fill(double(index),cnorm); 
	    }
         }

         int ieta = i - 42;        // -41 -1, 0 40 
         if(ieta >=0 ) ieta +=1;   // -41 -1, 1 41  - to make it detector-like

         if(ieta >= -20 && ieta <= 20 )
	    {phi_factor = 72.;}
         else {
	    if(ieta >= 40 || ieta <= -40 ) {phi_factor = 18.;}
         else 
	   phi_factor = 36.;
         }  
         if(ieta >= 0) ieta -= 1; // -41 -1, 0 40  - to bring back to histo num
	       
         /*
         std::cout << "*** ieta = " << ieta << "  sumphi_hb1, sumphi_hb2, sumphi_he1, sumphi_he2, simphi_he3, sumphi_ho, simphi_hf1, sumphi_hf2" << std::endl 
		<< sumphi_hb1 << " " << sumphi_hb2 << " " << sumphi_he1 << " "
		<< sumphi_he2 << " " << simphi_he3 << " " << sumphi_ho  << " " 
		<< simphi_hf1 << " " << sumphi_hf2 << std::endl << std::endl;
         */
         //Occupancy vs. ieta histos are drawn, RMS is not
         cnorm = sumphi_hb1 / phi_factor;
         occupancy_vs_ieta_HB1->Fill(float(ieta), cnorm);
         cnorm = sumphi_hb2 / phi_factor;
         occupancy_vs_ieta_HB2->Fill(float(ieta), cnorm);

         if (doSLHC_){
            cnorm = sumphi_hb3 / phi_factor;
            occupancy_vs_ieta_HB3->Fill(float(ieta), cnorm);
            cnorm = sumphi_hb4 / phi_factor;
            occupancy_vs_ieta_HB4->Fill(float(ieta), cnorm);
            cnorm = sumphi_hb5 / phi_factor;
            occupancy_vs_ieta_HB5->Fill(float(ieta), cnorm);
            cnorm = sumphi_hb6 / phi_factor;
            occupancy_vs_ieta_HB6->Fill(float(ieta), cnorm);
            cnorm = sumphi_hb7 / phi_factor;
            occupancy_vs_ieta_HB7->Fill(float(ieta), cnorm);
         }

         cnorm = sumphi_he1 / phi_factor;
         occupancy_vs_ieta_HE1->Fill(float(ieta), cnorm);
         cnorm = sumphi_he2 / phi_factor;
         occupancy_vs_ieta_HE2->Fill(float(ieta), cnorm);
         cnorm = sumphi_he3 / phi_factor;
         occupancy_vs_ieta_HE3->Fill(float(ieta), cnorm);

         if (doSLHC_){
            cnorm = sumphi_he4 / phi_factor;
            occupancy_vs_ieta_HE4->Fill(float(ieta), cnorm);
            cnorm = sumphi_he5 / phi_factor;
            occupancy_vs_ieta_HE5->Fill(float(ieta), cnorm);
            cnorm = sumphi_he6 / phi_factor;
            occupancy_vs_ieta_HE6->Fill(float(ieta), cnorm);
            cnorm = sumphi_he7 / phi_factor;
            occupancy_vs_ieta_HE7->Fill(float(ieta), cnorm);
         }

         cnorm = sumphi_ho / phi_factor;
         occupancy_vs_ieta_HO->Fill(float(ieta), cnorm);
         cnorm = sumphi_hf1 / phi_factor;
         occupancy_vs_ieta_HF1->Fill(float(ieta), cnorm);
         cnorm = sumphi_hf2 / phi_factor;
         occupancy_vs_ieta_HF2->Fill(float(ieta), cnorm);
      
         if (useAllHistos){
	  // RMS vs ieta (Emean's one)
	    cnorm = emean_vs_ieta_HB1->getBinError(i);
	    RMS_vs_ieta_HB1->Fill(ieta,cnorm);
	    cnorm = emean_vs_ieta_HB2->getBinError(i);
	    RMS_vs_ieta_HB2->Fill(ieta,cnorm);

            cnorm = emean_vs_ieta_HB3->getBinError(i);
            RMS_vs_ieta_HB3->Fill(ieta,cnorm);
            cnorm = emean_vs_ieta_HB4->getBinError(i);
            RMS_vs_ieta_HB4->Fill(ieta,cnorm);
            cnorm = emean_vs_ieta_HB5->getBinError(i);
            RMS_vs_ieta_HB5->Fill(ieta,cnorm);
            cnorm = emean_vs_ieta_HB6->getBinError(i);
            RMS_vs_ieta_HB6->Fill(ieta,cnorm);
            cnorm = emean_vs_ieta_HB7->getBinError(i);
            RMS_vs_ieta_HB7->Fill(ieta,cnorm);

	    cnorm = emean_vs_ieta_HE1->getBinError(i);
	    RMS_vs_ieta_HE1->Fill(ieta,cnorm);
	    cnorm = emean_vs_ieta_HE2->getBinError(i);
	    RMS_vs_ieta_HE2->Fill(ieta,cnorm);
	    cnorm = emean_vs_ieta_HE3->getBinError(i);
	    RMS_vs_ieta_HE3->Fill(ieta,cnorm);

            cnorm = emean_vs_ieta_HE4->getBinError(i);
            RMS_vs_ieta_HE4->Fill(ieta,cnorm);
            cnorm = emean_vs_ieta_HE5->getBinError(i);
            RMS_vs_ieta_HE5->Fill(ieta,cnorm);
            cnorm = emean_vs_ieta_HE6->getBinError(i);
            RMS_vs_ieta_HE6->Fill(ieta,cnorm);
            cnorm = emean_vs_ieta_HE7->getBinError(i);
            RMS_vs_ieta_HE7->Fill(ieta,cnorm);


	    cnorm = emean_vs_ieta_HB1->getBinError(i);
	    RMS_vs_ieta_HO->Fill(ieta,cnorm);
	    cnorm = emean_vs_ieta_HB1->getBinError(i);
	    RMS_vs_ieta_HF1->Fill(ieta,cnorm);
	    cnorm = emean_vs_ieta_HB1->getBinError(i);
	    RMS_vs_ieta_HF2->Fill(ieta,cnorm);
         }
      }  // end of i-loop

    
      // RMS seq (not drawn)
      if(useAllHistos){
         nx = emean_seqHB1->getNbinsX();    
         for(int ibin = 1; ibin <= nx; ibin++ ){
	    cnorm = emean_seqHB1->getBinError(ibin);
	    RMS_seq_HB1->setBinContent(ibin, cnorm);
	    cnorm = emean_seqHB2->getBinError(ibin);
	    RMS_seq_HB2->setBinContent(ibin, cnorm);
            cnorm = emean_seqHB3->getBinError(ibin);
            RMS_seq_HB3->setBinContent(ibin, cnorm);
            cnorm = emean_seqHB4->getBinError(ibin);
            RMS_seq_HB4->setBinContent(ibin, cnorm);
            cnorm = emean_seqHB5->getBinError(ibin);
            RMS_seq_HB5->setBinContent(ibin, cnorm);
            cnorm = emean_seqHB6->getBinError(ibin);
            RMS_seq_HB6->setBinContent(ibin, cnorm);
            cnorm = emean_seqHB7->getBinError(ibin);
            RMS_seq_HB7->setBinContent(ibin, cnorm);


	    cnorm = emean_seqHO->getBinError(ibin);
	    RMS_seq_HO->setBinContent(ibin, cnorm);
         }
         nx = emean_seqHE1->getNbinsX();    
         for(int ibin = 1; ibin <= nx; ibin++ ){
	    cnorm = emean_seqHE1->getBinError(ibin);
	    RMS_seq_HE1->setBinContent(ibin, cnorm);
	    cnorm = emean_seqHE2->getBinError(ibin);
	    RMS_seq_HE2->setBinContent(ibin, cnorm);
	    cnorm = emean_seqHE3->getBinError(ibin);
	    RMS_seq_HE3->setBinContent(ibin, cnorm);
            cnorm = emean_seqHE4->getBinError(ibin);
            RMS_seq_HE4->setBinContent(ibin, cnorm);
            cnorm = emean_seqHE5->getBinError(ibin);
            RMS_seq_HE5->setBinContent(ibin, cnorm);
            cnorm = emean_seqHE6->getBinError(ibin);
            RMS_seq_HE6->setBinContent(ibin, cnorm);
            cnorm = emean_seqHE7->getBinError(ibin);
            RMS_seq_HE7->setBinContent(ibin, cnorm);


         }
         nx = emean_seqHF1->getNbinsX();    
         for(int ibin = 1; ibin <= nx; ibin++ ){
	    cnorm = emean_seqHF1->getBinError(ibin);
	    RMS_seq_HF1->setBinContent(ibin, cnorm);
	    cnorm = emean_seqHF2->getBinError(ibin);
	    RMS_seq_HF2->setBinContent(ibin, cnorm);
         }
      }
      //Status Word (drawn)
      nx = RecHit_StatusWord_HB->getNbinsX();    
      for (int ibin = 1;  ibin <= nx; ibin++) {
         cnorm = RecHit_StatusWord_HB->getBinContent(ibin) / (fev * 2592.);
         RecHit_StatusWord_HB->setBinContent(ibin,cnorm);
      
         cnorm = RecHit_StatusWord_HE->getBinContent(ibin) / (fev * 2592.);
         RecHit_StatusWord_HE->setBinContent(ibin,cnorm);
      
         cnorm = RecHit_StatusWord_HO->getBinContent(ibin) / (fev * 2160.);
         RecHit_StatusWord_HO->setBinContent(ibin,cnorm);
      
         cnorm = RecHit_StatusWord_HF->getBinContent(ibin) / (fev * 1728.);
         RecHit_StatusWord_HF->setBinContent(ibin,cnorm);

         cnorm = RecHit_Aux_StatusWord_HB->getBinContent(ibin) / (fev * 2592.);
         RecHit_Aux_StatusWord_HB->setBinContent(ibin,cnorm);
      
         cnorm = RecHit_Aux_StatusWord_HE->getBinContent(ibin) / (fev * 2592.);
         RecHit_Aux_StatusWord_HE->setBinContent(ibin,cnorm);
      
         cnorm = RecHit_Aux_StatusWord_HO->getBinContent(ibin) / (fev * 2160.);
         RecHit_Aux_StatusWord_HO->setBinContent(ibin,cnorm);
      
         cnorm = RecHit_Aux_StatusWord_HF->getBinContent(ibin) / (fev * 1728.);
         RecHit_Aux_StatusWord_HF->setBinContent(ibin,cnorm);
      }
      //HF 2-bit status word (not drawn)
      if(useAllHistos){
         nx = RecHit_StatusWord_HF67->getNbinsX();    
         for (int ibin = 1;  ibin <= nx; ibin++) {
	    cnorm = RecHit_StatusWord_HF67->getBinContent(ibin) / (fev * 1728.);
	    RecHit_StatusWord_HF67->setBinContent(ibin,cnorm);
         }
      }
   }

   return 1;
}

DEFINE_FWK_MODULE(HcalRecHitsClient);
