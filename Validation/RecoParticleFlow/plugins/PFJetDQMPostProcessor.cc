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
  //for(unsigned int ipt = 0; ipt < ptBins.size(); ++ipt) std::cout << ptBins[ipt] << std::endl;
  
  std::string stitle;
  char ctitle[50];
  std::vector<MonitorElement*> vME_presponse;
  std::vector<MonitorElement*> vME_preso;
  std::vector<MonitorElement*> vME_preso_rms;

  MonitorElement* me;
  TH1F * h_resp;
  TH1F * h_genjet_pt;

  //
  // Response distributions
  //
  for(unsigned int ieta = 1; ieta < etaBins.size(); ++ieta) {

    stitle = genjetDir + "genjet_pt" + "_eta" + seta(etaBins[ieta]); 
    me=iget_.get(stitle);
    h_genjet_pt = (TH1F*) me->getTH1F();
          
    stitle = "presponse_eta"+seta(etaBins[ieta]);
    sprintf(ctitle,"Jet pT response, %4.1f<|#eta|<%4.1f",etaBins[ieta],etaBins[ieta+1]);
    TH1F *h_presponse = new TH1F(stitle.c_str(),ctitle,nPtBins,ptBinsArray);

    stitle = "preso_eta"+seta(etaBins[ieta]);
    sprintf(ctitle,"Jet pT resolution, %4.1f<|#eta|<%4.1f",etaBins[ieta],etaBins[ieta+1]);
    TH1F *h_preso = new TH1F(stitle.c_str(),ctitle,nPtBins,ptBinsArray);

    stitle = "preso_eta"+seta(etaBins[ieta])+"_rms";
    sprintf(ctitle,"Jet pT resolution using RMS, %4.1f<|#eta|<%4.1f",etaBins[ieta],etaBins[ieta+1]);
    TH1F *h_preso_rms = new TH1F(stitle.c_str(),ctitle,nPtBins,ptBinsArray);

    for(unsigned int ipt = 0; ipt < ptBins.size()-1; ++ipt) {

      stitle = jetResponseDir + "reso_dist_" + spt(ptBins[ipt],ptBins[ipt+1]) + "_eta" + seta(etaBins[ieta]); 
      me=iget_.get(stitle);
      h_resp = (TH1F*) me->getTH1F();

      // Fit-based
      double resp=1.0, resp_err=0.0, reso=0.0, reso_err=0.0;
      fitResponse(h_resp, h_genjet_pt, ptBins[ipt], recoptcut,
		  resp, resp_err, reso, reso_err);
      
      h_presponse->SetBinContent(ipt+1,resp);
      h_presponse->SetBinError(ipt+1,resp_err);
      h_preso->SetBinContent(ipt+1,reso);
      h_preso->SetBinError(ipt+1,reso_err);
      
      // RMS-based
      double std = h_resp->GetStdDev();
      double std_error = h_resp->GetStdDevError();

      // Scale each bin with mean response
      double mean = 1.0;
      double mean_error = 0.0;
      double err = 0.0;
      if (h_resp->GetMean()>0){
	mean = h_resp->GetMean();
	mean_error = h_resp->GetMeanError();
        if (std > 0.0 && mean > 0.0)
	  err = std/mean * sqrt(pow(std_error,2) / pow(std,2) + pow(mean_error,2) / pow(mean,2));
	if (mean > 0.0)
	  std /= mean;
      }

      h_preso_rms->SetBinContent(ipt+1,std);
      h_preso_rms->SetBinError(ipt+1,err);
	
    } // ipt

    stitle = "presponse_eta"+seta(etaBins[ieta]);
    me = ibook_.book1D(stitle.c_str(),h_presponse);
    vME_presponse.push_back(me);

    stitle = "preso_eta"+seta(etaBins[ieta]);
    me = ibook_.book1D(stitle.c_str(),h_preso);
    vME_preso.push_back(me);

    stitle = "preso_eta"+seta(etaBins[ieta])+"_rms";
    me = ibook_.book1D(stitle.c_str(),h_preso_rms);
    vME_preso_rms.push_back(me);
    
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
  
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetDQMPostProcessor);
