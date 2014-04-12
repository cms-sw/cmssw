{
gROOT->Reset();
#include "Riostream.h"

 TFile * theFile1 = new TFile("gensimLevelPlots.root");
 std::cout<<"Reading histos"<<std::endl;

 TH1F *  particleEta           = (TH1F*) (theFile1->Get("hscpValidator/particleEta"));
 TH1F *  particlePhi           = (TH1F*) (theFile1->Get("hscpValidator/particlePhi"));
 TH1F *  particleP             = (TH1F*) (theFile1->Get("hscpValidator/particleP"));
 TH1F *  particlePt            = (TH1F*) (theFile1->Get("hscpValidator/particlePt"));
 TH1F *  particleMass          = (TH1F*) (theFile1->Get("hscpValidator/particleMass"));
 TH1F *  particleStatus        = (TH1F*) (theFile1->Get("hscpValidator/particleStatus"));
 TH1F *  particleBeta          = (TH1F*) (theFile1->Get("hscpValidator/particleBeta"));
 TH1F *  particleBetaInverse   = (TH1F*) (theFile1->Get("hscpValidator/particleBetaInverse"));

 std::string command = "mkdir gen";
 system(command.c_str());

 TCanvas t;
 t.cd();
  
 particleEta->Draw("E");
 t.Print("gen/eta.png");
 particlePhi->Rebin(4);
 particlePhi->Draw("E");
 t.Print("gen/phi.png");
 particlePt->Draw();
 t.Print("gen/pt.png");
 particleP->Draw();
 t.Print("gen/p.png");
 particleBeta->Draw("E");
 t.Print("gen/beta.png");
 particleBetaInverse->Draw();
 t.Print("gen/1beta.png");

 TH1F *   simHitsEcalEnergyHistEB_                  = (TH1F*) (theFile1->Get("hscpValidator/ecalEnergyOfSimHitsEB"));		   
 TH1F *   simHitsEcalEnergyHistEE_		    = (TH1F*) (theFile1->Get("hscpValidator/ecalEnergyOfSimHitsEE"));		   
 TH1F *   simHitsEcalTimeHistEB_		    = (TH1F*) (theFile1->Get("hscpValidator/ecalTimingOfSimHitsEB"));		   		   
 TH1F *   simHitsEcalTimeHistEE_		    = (TH1F*) (theFile1->Get("hscpValidator/ecalTimingOfSimHitsEE"));		   		   
 TH1F *   simHitsEcalNumHistEB_			    = (TH1F*) (theFile1->Get("hscpValidator/ecalNumberOfSimHitsEB"));		   		   
 TH1F *   simHitsEcalNumHistEE_			    = (TH1F*) (theFile1->Get("hscpValidator/ecalNumberOfSimHitsEE"));		   		   
 TH2F *   simHitsEcalEnergyVsTimeHistEB_	    = (TH2F*) (theFile1->Get("hscpValidator/ecalEnergyVsTimeOfSimHitsEB"));		   	   
 TH2F *   simHitsEcalEnergyVsTimeHistEE_	    = (TH2F*) (theFile1->Get("hscpValidator/ecalEnergyVsTimeOfSimHitsEE"));		   	   
 TH1F *    simHitsEcalDigiMatchEnergyHistEB_        = (TH1F*) (theFile1->Get("hscpValidator/ecalEnergyOfDigiMatSimHitsEB"));		   	   
 TH1F *   simHitsEcalDigiMatchEnergyHistEE_	    = (TH1F*) (theFile1->Get("hscpValidator/ecalEnergyOfDigiMatSimHitsEE"));		   	   
 TH1F *   simHitsEcalDigiMatchTimeHistEB_	    = (TH1F*) (theFile1->Get("hscpValidator/ecalTimingOfDigiMatSimHitsEB"));		   	   
 TH1F *   simHitsEcalDigiMatchTimeHistEE_	    = (TH1F*) (theFile1->Get("hscpValidator/ecalTimingOfDigiMatSimHitsEE"));		   	   
 TH2F *   simHitsEcalDigiMatchEnergyVsTimeHistEB_   = (TH2F*) (theFile1->Get("hscpValidator/ecalEnergyVsTimeOfDigiMatSimHitsEB"));		    
 TH2F *   simHitsEcalDigiMatchEnergyVsTimeHistEE_   = (TH2F*) (theFile1->Get("hscpValidator/ecalEnergyVsTimeOfDigiMatSimHitsEE"));		    
 TH1F *   simHitsEcalDigiMatchIEtaHist_		    = (TH1F*) (theFile1->Get("hscpValidator/ecalIEtaOfDigiMatchSimHits"));		   	   
 TH1F *   simHitsEcalDigiMatchIPhiHist_		    = (TH1F*) (theFile1->Get("hscpValidator/ecalIPhiOfDigiMatchSimHits"));		   	   
 TH1F *    digisEcalNumHistEB_                      = (TH1F*) (theFile1->Get("hscpValidator/ecalDigisNumberEB"));		   		   
 TH1F *   digisEcalNumHistEE_			    = (TH1F*) (theFile1->Get("hscpValidator/ecalDigisNumberEE"));		   		   
 TH2F *   digiOccupancyMapEB_			    = (TH2F*) (theFile1->Get("hscpValidator/ecalDigiOccupancyMapEB"));		   		   
 TH2F *   digiOccupancyMapEEP_			    = (TH2F*) (theFile1->Get("hscpValidator/ecalDigiOccupancyMapEEM"));		   		   
 TH2F *   digiOccupancyMapEEM_			    = (TH2F*) (theFile1->Get("hscpValidator/ecalDigiOccupancyMapEEP"));		               

 std::string command = "mkdir sim";
 system(command.c_str());
 std::string command = "mkdir digi";
 system(command.c_str());


 TStyle* thisStyle = new TStyle("myStyle", "NewStyle");
  thisStyle->SetOptStat(2222211);
  thisStyle->SetPalette(1);
  thisStyle->cd();

 system(command.c_str());

 t.cd();

  simHitsEcalEnergyHistEB_->Draw();
  t.Print("sim/simHitsEnergyEB.png");
  simHitsEcalEnergyHistEE_->Draw();
  t.Print("sim/simHitsEnergyEE.png");
  simHitsEcalTimeHistEB_->Draw();
  t.Print("sim/simHitsTimeEB.png");
  simHitsEcalTimeHistEE_->Draw();
  t.Print("sim/simHitsTimeEE.png");
  simHitsEcalNumHistEB_->Draw();
  t.Print("sim/simHitsNumberEB.png");
  simHitsEcalNumHistEE_->Draw();
  t.Print("sim/simHitsNumberEE.png");
  simHitsEcalEnergyVsTimeHistEB_->Draw("colz");
  t.Print("sim/simHitsEnergyVsTimeEB.png");
  simHitsEcalEnergyVsTimeHistEE_->Draw("colz");
  t.Print("sim/simHitsEnergyVsTimeEE.png");
  
  simHitsEcalDigiMatchEnergyHistEB_->Draw();
  t.Print("sim/digiMatchedSimHitsEnergyEB.png");
  simHitsEcalDigiMatchEnergyHistEE_->Draw();
  t.Print("sim/digiMatchedSimHitsEnergyEE.png");
  simHitsEcalDigiMatchTimeHistEB_->Draw();
  t.Print("sim/digiMatchedSimHitsTimeEB.png");
  simHitsEcalDigiMatchTimeHistEE_->Draw();
  t.Print("sim/digiMatchedSimHitsTimeEE.png");
  simHitsEcalDigiMatchEnergyVsTimeHistEB_->Draw("colz");
  t.Print("sim/digiMatchedSimHitsEnergyVsTimeEB.png");
  simHitsEcalDigiMatchEnergyVsTimeHistEE_->Draw("colz");
  t.Print("sim/digiMatchedSimHitsEnergyVsTimeEE.png");

  simHitsEcalDigiMatchIEtaHist_->Draw();
  t.Print("sim/digiMatchedSimHitsIeta.png");
  simHitsEcalDigiMatchIPhiHist_->Draw();
  t.Print("sim/digiMatchedSimHitsIphi.png");

  digisEcalNumHistEB_->Draw();
  t.Print("sim/numDigisEB.png");
  digisEcalNumHistEE_->Draw();
  t.Print("sim/numDigisEE.png");

  thisStyle->SetOptStat(11);
  thisStyle->cd();

  digiOccupancyMapEB_->Draw("colz");
  t.Print("digi/digiOccupancyEB.png");
  digiOccupancyMapEEP_->Draw("colz");
  t.Print("digi/digiOccupancyEEP.png");
  digiOccupancyMapEEM_->Draw("colz");
  t.Print("digi/digiOccupancyEEM.png");



  //The RPC Part


  TH1F* residualsRPCRecHitSimDigis_;
  TH1F* efficiencyRPCRecHitSimDigis_;
  TH1F* cluSizeDistribution_; 
  TH1F* rpcTimeOfFlightBarrel_[6];       
  TH1F* rpcBXBarrel_[6];       
  TH1F* rpcTimeOfFlightEndCap_[3];       
  TH1F* rpcBXEndCap_[3];    



residualsRPCRecHitSimDigis_ = (TH1F*) (theFile1->Get("hscpValidator/residualsRPCRecHitSimDigis"));
efficiencyRPCRecHitSimDigis_= (TH1F*) (theFile1->Get("hscpValidator/efficiencyRPCRecHitSimDigis"));
cluSizeDistribution_	    = (TH1F*) (theFile1->Get("hscpValidator/RPCCluSizeDistro"));
rpcTimeOfFlightBarrel_[0]   = (TH1F*) (theFile1->Get("hscpValidator/RPCToFLayer1"));
rpcTimeOfFlightBarrel_[1]   = (TH1F*) (theFile1->Get("hscpValidator/RPCToFLayer2"));
rpcTimeOfFlightBarrel_[2]   = (TH1F*) (theFile1->Get("hscpValidator/RPCToFLayer3"));
rpcTimeOfFlightBarrel_[3]   = (TH1F*) (theFile1->Get("hscpValidator/RPCToFLayer4"));
rpcTimeOfFlightBarrel_[4]   = (TH1F*) (theFile1->Get("hscpValidator/RPCToFLayer5"));
rpcTimeOfFlightBarrel_[5]   = (TH1F*) (theFile1->Get("hscpValidator/RPCToFLayer6"));
rpcBXBarrel_[0]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXLayer1"));
rpcBXBarrel_[1]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXLayer2"));
rpcBXBarrel_[2]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXLayer3"));
rpcBXBarrel_[3]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXLayer4"));
rpcBXBarrel_[4]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXLayer5"));
rpcBXBarrel_[5]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXLayer6"));
rpcTimeOfFlightEndCap_[0]  = (TH1F*) (theFile1->Get("hscpValidator/RPCToFDisk1"));
rpcTimeOfFlightEndCap_[1]  = (TH1F*) (theFile1->Get("hscpValidator/RPCToFDisk2"));
rpcTimeOfFlightEndCap_[2]  = (TH1F*) (theFile1->Get("hscpValidator/RPCToFDisk3"));
rpcBXEndCap_[0]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXDisk1"));
rpcBXEndCap_[1]		    = (TH1F*) (theFile1->Get("hscpValidator/RPCBXDisk2"));
rpcBXEndCap_[2]             = (TH1F*) (theFile1->Get("hscpValidator/RPCBXDisk3"));

residualsRPCRecHitSimDigis_->Draw();
  t.Print("digi/rpcresiduals.png");

  efficiencyRPCRecHitSimDigis_->Draw();
  t.Print("digi/rpcefficiency.png");
  
  cluSizeDistribution_->Draw();
  t.Print("digi/cluRPCSize.png");

  TLegend *leg = new TLegend(0.6,0.85,0.9,0.3);
  rpcTimeOfFlightBarrel_[0]->SetFillColor(1);
  rpcTimeOfFlightBarrel_[0]->Draw();
  rpcTimeOfFlightBarrel_[0]->GetXaxis()->SetTitle("ToF (ns)");
  std::stringstream legend;
  legend.str("");
  legend<<"ToF Layer 1 Mean "<<rpcTimeOfFlightBarrel_[0]->GetMean()<<"ns";
  leg->AddEntry(rpcTimeOfFlightBarrel_[0],legend.str().c_str(),"f");
  float max = rpcTimeOfFlightBarrel_[0]->GetBinContent(rpcTimeOfFlightBarrel_[0]->GetMaximumBin());
  for(int i=1; i<6;i++){
    rpcTimeOfFlightBarrel_[i]->SetFillColor(i+1);
    legend.str("");
    legend<<"ToF Layer "<<i+1<<" Mean "<<rpcTimeOfFlightBarrel_[i]->GetMean()<<"ns";
    leg->AddEntry(rpcTimeOfFlightBarrel_[i],legend.str().c_str(),"f");
    rpcTimeOfFlightBarrel_[i]->Draw("same"); 
    float thismax = rpcTimeOfFlightBarrel_[i]->GetBinContent(rpcTimeOfFlightBarrel_[i]->GetMaximumBin());
    if(thismax > max) max = thismax;
    cout<<"max "<<max<<endl;
  }

  for(int i=0; i<6;i++) rpcTimeOfFlightBarrel_[i]->SetMaximum(max);

  leg->Draw("same");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  t.Print("sim/RPCBToF.png");
  
  leg->Clear();
  
  rpcTimeOfFlightEndCap_[2]->Draw();
  rpcTimeOfFlightEndCap_[2]->SetFillColor(3);
  rpcTimeOfFlightEndCap_[2]->GetXaxis()->SetTitle("ToF (ns)");
  
  legend.str("");
  legend<<"ToF Disk 3 Mean "<<rpcTimeOfFlightEndCap_[2]->GetMean()<<"ns";
  leg->AddEntry(rpcTimeOfFlightEndCap_[2],legend.str().c_str(),"f");

  float max = rpcTimeOfFlightEndCap_[2]->GetBinContent(rpcTimeOfFlightEndCap_[2]->GetMaximumBin());
  cout<<"max "<<max<<endl;
  
  for(int i=1;i>=0;i--){
    rpcTimeOfFlightEndCap_[i]->SetFillColor(i+1);
    legend.str("");
    legend<<"ToF Disk "<<i+1<<" Mean "<<rpcTimeOfFlightEndCap_[i]->GetMean()<<"ns";
    leg->AddEntry(rpcTimeOfFlightEndCap_[i],legend.str().c_str(),"f");
    rpcTimeOfFlightEndCap_[i]->Draw("same");
    float thismax = rpcTimeOfFlightEndCap_[i]->GetBinContent(rpcTimeOfFlightEndCap_[i]->GetMaximumBin());
    if(thismax > max) max = thismax;
    cout<<"max "<<max<<endl;
  }
  
  rpcTimeOfFlightEndCap_[0]->SetMaximum(max);
  rpcTimeOfFlightEndCap_[1]->SetMaximum(max);
  rpcTimeOfFlightEndCap_[2]->SetMaximum(max);
  

  leg->Draw("same");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  t.Print("sim/RPCEToF.png");
  
  leg->Clear();
  
  rpcBXBarrel_[0]->SetLineColor(1);
  rpcBXBarrel_[0]->Draw();
  rpcBXBarrel_[0]->GetXaxis()->SetTitle("BX Units(25ns)");
  rpcBXBarrel_[0]->SetLineWidth(3);
  
  legend.str("");
  legend<<"BX Layer 1 Mean "<<rpcBXBarrel_[0]->GetMean();
  leg->AddEntry(rpcBXBarrel_[0],legend.str().c_str(),"l");

  float max = rpcBXBarrel_[0]->GetBinContent(rpcBXBarrel_[0]->GetMaximumBin());
				
  for(int i=1; i<6;i++){
    rpcBXBarrel_[i]->SetLineColor(i+1);
    legend.str("");
    legend<<"BX Layer "<<i+1<<" Mean "<<rpcBXBarrel_[i]->GetMean();
    leg->AddEntry(rpcBXBarrel_[i],legend.str().c_str(),"l");
    rpcBXBarrel_[i]->SetLineWidth(3);
    rpcBXBarrel_[i]->Draw("same");
    float thismax = rpcBXBarrel_[i]->GetBinContent(rpcBXBarrel_[i]->GetMaximumBin());
    if(thismax > max) max = thismax;
  }
  
  for(int i=0; i<6;i++)rpcBXBarrel_[i]->SetMaximum(max);
  
  leg->Draw("same");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  
  t.Print("digi/RPCBBX.png");

  leg->Clear();
  
  rpcBXEndCap_[0]->SetLineColor(1);
  rpcBXEndCap_[0]->Draw();
  rpcBXEndCap_[0]->GetXaxis()->SetTitle("BX Units(25ns)");
  rpcBXEndCap_[0]->SetLineWidth(3);
  
  legend.str("");
  legend<<"BX Disk 1 Mean "<<rpcBXEndCap_[0]->GetMean();
  leg->AddEntry(rpcBXEndCap_[0],legend.str().c_str(),"l");

  float max = rpcBXEndCap_[0]->GetBinContent(rpcBXEndCap_[0]->GetMaximumBin());
 
  for(int i=1; i<3;i++){
    rpcBXEndCap_[i]->SetLineColor(i+1);
    legend.str("");
    legend<<"BX Disk "<<i+1<<" Mean "<<rpcBXEndCap_[i]->GetMean();
    leg->AddEntry(rpcBXEndCap_[i],legend.str().c_str(),"l");
    rpcBXEndCap_[i]->SetLineWidth(3);
    rpcBXEndCap_[i]->Draw("same");
    float thismax = rpcBXEndCap_[i]->GetBinContent(rpcBXEndCap_[i]->GetMaximumBin());
    if(thismax > max) max = thismax;
  }

  for(int i=0; i<3;i++)rpcBXEndCap_[i]->SetMaximum(max);

  leg->Draw("same");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  
  t.Print("digi/RPCEBX.png");

  exit(0);

}









