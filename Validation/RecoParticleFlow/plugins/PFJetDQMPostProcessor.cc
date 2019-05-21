// -*- C++ -*-
//
// Package:    Validation/RecoParticleFlow
// Class:      PFJetDQMPostProcessor.cc
// 
// Main Developer:  "Juska Pekkanen"
// Original Author:  "Kenichi Hatakeyama"
//

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class decleration
//

class PFJetDQMPostProcessor : public DQMEDHarvester {
   public:
      explicit PFJetDQMPostProcessor(const edm::ParameterSet&);
      ~PFJetDQMPostProcessor() override;

   private:
      void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override ;
      void fitResponse(TH1F* hreso, TH1F* h_genjet_pt, double ptlow, double recoptcut,
		       double& resp, double& resp_err, double& reso, double& reso_err);
  
      std::string jetResponseDir;
      std::string genjetDir;
      std::vector<double> ptBins;
      std::vector<double> etaBins;

      double recoptcut;
  
      bool debug=false;

      std::string seta(double eta){
	std::string seta = std::to_string(int(eta*10.));
	if (seta.length()<2) seta = "0"+seta;
	return seta;
      }
  
      std::string spt(double ptmin,double ptmax){
	std::string spt = std::to_string(int(ptmin)) + "_" + std::to_string(int(ptmax));
	return spt;
      }
  
};

// Some switches
//
// constructors and destructor
//
PFJetDQMPostProcessor::PFJetDQMPostProcessor(const edm::ParameterSet& iConfig)
{

  jetResponseDir = iConfig.getParameter<std::string>("jetResponseDir");
  genjetDir = iConfig.getParameter<std::string>("genjetDir");
  ptBins = iConfig.getParameter< std::vector<double> >("ptBins");
  etaBins = iConfig.getParameter< std::vector<double> >("etaBins");
  recoptcut = iConfig.getParameter< double >("recoPtCut");

}


PFJetDQMPostProcessor::~PFJetDQMPostProcessor()
{ 
}


// ------------ method called right after a run ends ------------
void 
PFJetDQMPostProcessor::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_)
{

  iget_.setCurrentFolder(jetResponseDir);

  double ptBinsArray[ptBins.size()];
  unsigned int nPtBins = ptBins.size()-1;
  std::copy(ptBins.begin(),ptBins.end(),ptBinsArray);
  for(unsigned int ipt = 0; ipt < ptBins.size(); ++ipt) std::cout << ptBins[ipt] << std::endl;
  
  std::string stitle;
  std::vector<MonitorElement*> vME_presponse;
  std::vector<MonitorElement*> vME_preso;
  std::vector<MonitorElement*> vME_preso_rms;

  MonitorElement* mtmp;
  TH1F * htmp;
  TH1F * htmp2;

  //
  // Response distributions
  //
  for(unsigned int ieta = 1; ieta < etaBins.size(); ++ieta) {

    stitle = "presponse_eta"+seta(etaBins[ieta]);
    htmp = new TH1F(stitle.c_str(),stitle.c_str(),nPtBins,ptBinsArray);
    mtmp = ibook_.book1D(stitle.c_str(),htmp);
    vME_presponse.push_back(mtmp);

    stitle = "preso_eta"+seta(etaBins[ieta]);
    htmp = new TH1F(stitle.c_str(),stitle.c_str(),nPtBins,ptBinsArray);
    mtmp = ibook_.book1D(stitle.c_str(),htmp);
    vME_preso.push_back(mtmp);

    stitle = "preso_eta"+seta(etaBins[ieta])+"_rms";
    htmp = new TH1F(stitle.c_str(),stitle.c_str(),nPtBins,ptBinsArray);
    mtmp = ibook_.book1D(stitle.c_str(),htmp);
    vME_preso_rms.push_back(mtmp);

  }
  
  //
  // Response distributions
  //
  for(unsigned int ieta = 1; ieta < etaBins.size(); ++ieta) {

    stitle = genjetDir + "genjet_pt" + "_eta" + seta(etaBins[ieta]); 
    //std::cout << stitle << std::endl;
    mtmp=iget_.get(stitle);
    htmp2 = (TH1F*) mtmp->getTH1F();
    //htmp2->Print();
          
    stitle = "presponse_eta"+seta(etaBins[ieta]);
    TH1F *h_presponse = new TH1F(stitle.c_str(),stitle.c_str(),nPtBins,ptBinsArray);

    stitle = "preso_eta"+seta(etaBins[ieta]);
    TH1F *h_preso = new TH1F(stitle.c_str(),stitle.c_str(),nPtBins,ptBinsArray);

    stitle = "preso_eta"+seta(etaBins[ieta])+"_rms";
    TH1F *h_preso_rms = new TH1F(stitle.c_str(),stitle.c_str(),nPtBins,ptBinsArray);

    for(unsigned int ipt = 0; ipt < ptBins.size()-1; ++ipt) {

      stitle = jetResponseDir + "reso_dist_" + spt(ptBins[ipt],ptBins[ipt+1]) + "_eta" + seta(etaBins[ieta]); 
      //std::cout << stitle << std::endl;
      mtmp=iget_.get(stitle);
      htmp = (TH1F*) mtmp->getTH1F();
      //htmp->Print();

      // Fit-based
      double resp=1.0, resp_err=0.0, reso=0.0, reso_err=0.0;
      fitResponse(htmp, htmp2, ptBins[ipt], recoptcut,
		  resp, resp_err, reso, reso_err);
      
      h_presponse->SetBinContent(ipt+1,resp);
      h_presponse->SetBinError(ipt+1,resp_err);
      h_preso->SetBinContent(ipt+1,reso);
      h_preso->SetBinError(ipt+1,reso_err);
      
      // RMS-based
      double std = htmp->GetStdDev();
      double std_error = htmp->GetStdDevError();

      // Scale each bin with mean response
      double mean = 1.0;
      double mean_error = 0.0;
      double err = 0.0;
      if (htmp->GetMean()>0){
	mean = htmp->GetMean();
	mean_error = htmp->GetMeanError();
        if (std > 0.0 && mean > 0.0)
	  err = std/mean * sqrt(pow(std_error,2) / pow(std,2) + pow(mean_error,2) / pow(mean,2));
	if (mean > 0.0)
	  std /= mean;
      }

      // std::cout << ptBins[ipt] << " " << etaBins[ieta] << " "
      // 		<< resp << " " << resp_err << " "
      // 		<< reso << " " << reso_err << " "
      // 		<< std << " " << err << std::endl;
	
      h_preso_rms->SetBinContent(ipt+1,std);
      h_preso_rms->SetBinError(ipt+1,err);
	
    } // ipt

    stitle = "presponse_eta"+seta(etaBins[ieta]);
    mtmp = ibook_.book1D(stitle.c_str(),h_presponse);
    vME_presponse.push_back(mtmp);

    stitle = "preso_eta"+seta(etaBins[ieta]);
    mtmp = ibook_.book1D(stitle.c_str(),h_preso);
    vME_preso.push_back(mtmp);

    stitle = "preso_eta"+seta(etaBins[ieta])+"_rms";
    mtmp = ibook_.book1D(stitle.c_str(),h_preso_rms);
    vME_preso_rms.push_back(mtmp);
    
  } // ieta

  //
  // Checks
  //
  if (debug){
    for(std::vector<MonitorElement*>::const_iterator i = vME_presponse.begin(); i != vME_presponse.end(); ++i)
      (*i)->getTH1F()->Print();
    for(std::vector<MonitorElement*>::const_iterator i = vME_preso.begin(); i != vME_preso.end(); ++i)
      (*i)->getTH1F()->Print();
    for(std::vector<MonitorElement*>::const_iterator i = vME_preso_rms.begin(); i != vME_preso_rms.end(); ++i)
      (*i)->getTH1F()->Print();
  }
  
}
void 
PFJetDQMPostProcessor::fitResponse(TH1F* hreso, TH1F* h_genjet_pt, double ptlow, double recoptcut,
				   double& resp, double& resp_err, double& reso, double& reso_err)
{

  double rmswidth = hreso->GetStdDev();
  double rmsmean = hreso->GetMean();
   
  double fitlow = rmsmean-1.5*rmswidth;   
  fitlow = TMath::Max(recoptcut/ptlow,fitlow);
  double fithigh = rmsmean+1.5*rmswidth;

  TF1 *fg = new TF1("fg","gaus",fitlow,fithigh);
  
  hreso->Fit("fg","RQ");

  resp     = fg->GetParameter(1);
  resp_err = fg->GetParError(1);
  reso     = fg->GetParameter(2);
  reso_err = fg->GetParError(2);

  // std::cout 
  // 	    << resp << " " << resp_err << " "
  // 	    << reso << " " << reso_err << std::endl;
  
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetDQMPostProcessor);
