#include "HcalMonitor.h"

using namespace TMath;

int monitor(char* input_file, char* output_file)
{

  TCanvas *c1 = new TCanvas();  
  TStyle *mystyle = new TStyle("mystyle","my style");
  initStyle(mystyle);
  mystyle->cd();

  index_map::index_map get_index;
  int true_index[4176];

  TFile* file=TFile::Open(input_file);
  if(!file) return 0;

  TTree* tree = (TTree*)file->Get("TPGntuple");
  if(!tree) return 0;

  int ieta[4176], iphi[4176], run, event, tpg_index[4176];
  float tpg_energy[4176], tpg_uncompressed[4176];
  float highest_energy[4176] = {0.0};
  float rec_energy[4167];
  //int eta, phi;

  //tree->SetBranchAddress("ieta",ieta);
  //tree->SetBranchAddress("iphi",iphi);
  tree->SetBranchAddress("run",&run);
  tree->SetBranchAddress("event",&event);
  tree->SetBranchAddress("tpg_energy",tpg_energy);
  // tree->SetBranchAddress("hit_energy",hit_energy);
  tree->SetBranchAddress("tpg_uncompressed",tpg_uncompressed);
  tree->SetBranchAddress("tpg_index",tpg_index);
  tree->SetBranchAddress("rec_energy",rec_energy);

  //2-D plots
  TH2F *fakes = new TH2F("fakes","Number of Fakes in Tower",65,-32,33,72,1,73);
  TH2F *nofire = new TH2F("nofire","Highest Energy with No TPG",65,-32,33,72,1,73);
  TH2F *hslope = new TH2F("hslope","Slope of Uncompressed TPG vs. Hit",65,-32,33,72,1,73);
  TH2F *res2D_mean = new TH2F("res2D_mean","Mean of (Hit - Uncompressed)/Hit",65,-32,33,72,1,73);
  TH2F *res2D_rms = new TH2F("res2D_rms","RMS of (Hit - Uncompressed)/Hit",65,-32,33,72,1,73);
  TH2F *slope_eta = new TH2F("slope_eta","Slope vs. iEta",33,0,32,100,0,5);
  TH2F *heffic = new TH2F("heffic","Fitted Efficiency",65,-32,33,72,1,73);
  TH2F *hthresh = new TH2F("hthresh","Fitted 50% Threshold",65,-32,33,72,1,73);
  TH2F *hwidth = new TH2F("hwidth","Fitted Turn-on Width",65,-32,33,72,1,73);
  TH2F *hit_tpg_lut1 = new TH2F("hit_tpg_lut1","Uncompressed E vs. Hit E: |ieta| < 21",80,0,100,80,0,100);
  TH2F *hit_tpg_lut2 = new TH2F("hit_tpg_lut2","Uncompressed E vs. Hit E: 21 <= |ieta| < 27",80,0,100,80,0,100);
  TH2F *hit_tpg_lut3 = new TH2F("hit_tpg_lut3","Uncompressed E vs. Hit E: 27 <= |ieta| < 29",80,0,100,80,0,100);
  TH2F *hit_tpg_lut4 = new TH2F("hit_tpg_lut4","Uncompressed E vs. Hit E: 29 <= |ieta|",80,0,100,80,0,100);
  TH2F *res_hit_lut1 = new TH2F("res_hit_lut1","(Hit E - Uncompressed E)/Hit E vs. Hit E: |ieta| < 21",80,0,100,12,-3,3);
  TH2F *res_hit_lut2 = new TH2F("res_hit_lut2","(Hit E - Uncompressed E)/Hit E vs. Hit E: 21 <= |ieta| < 27",80,0,100,12,-3,3);
  TH2F *res_hit_lut3 = new TH2F("res_hit_lut3","(Hit E - Uncompressed E)/Hit E vs. Hit E: 27 <= |ieta| < 29",80,0,100,12,-3,3);
  TH2F *res_hit_lut4 = new TH2F("res_hit_lut4","(Hit E - Uncompressed E)/Hit E vs. Hit E: 29 <= |ieta|",80,0,100,12,-3,3);

  //1-D plots
  TH1F *slope_sum = new TH1F("slope_sum","Summary of Slopes",100,0,2);
  TH1F *res1D_mean_lut1 = new TH1F("res1D_mean_lut1","Summary of Mean (Hit - Uncompressed)/Hit: |ieta| < 21",100,-0.2,0.2);
  TH1F *res1D_rms_lut1 = new TH1F("res1D_rms_lut1","Summary of RMS (Hit - Uncompressed)/Hit: |ieta| < 21",100,0,0.2);
  TH1F *res1D_mean_lut2 = new TH1F("res1D_mean_lut2","Summary of Mean (Hit - Uncompressed)/Hit: 21 <= |ieta| < 27",250,-0.5,0.5);
  TH1F *res1D_rms_lut2 = new TH1F("res1D_rms_lut2","Summary of RMS (Hit - Uncompressed)/Hit: 21 <= |ieta| < 27",100,0,0.2);
  TH1F *res1D_mean_lut3 = new TH1F("res1D_mean_lut3","Summary of Mean (Hit - Uncompressed)/Hit: 27 <= |ieta| < 29",250,-0.5,0.5);
  TH1F *res1D_rms_lut3 = new TH1F("res1D_rms_lut3","Summary of RMS (Hit - Uncompressed)/Hit: 27 <= |ieta| < 29",150,0,0.3);
  TH1F *res1D_mean_lut4 = new TH1F("res1D_mean_lut4","Summary of Mean (Hit - Uncompressed)/Hit: 29 <= |ieta|",300,-0.6,0.6);
  TH1F *res1D_rms_lut4 = new TH1F("res1D_rms_lut4","Summary of RMS (Hit - Uncompressed)/Hit: 29 <= |ieta|",150,0,0.3);
  TH1F *all_hits_lut1 = new TH1F("all_hits_lut1","all_hits: lut1",50,0,5);
  TH1F *all_hits_lut2 = new TH1F("all_hits_lut2","all_hits: lut2",50,0,5);
  TH1F *all_hits_lut3 = new TH1F("all_hits_lut3","all_hits: lut3",150,0,15);
  TH1F *all_hits_lut4 = new TH1F("all_hits_lut4","all_hits: lut4",300,5,30);
  TH1F *good_hits_lut1 = new TH1F("good_hits_lut1","good_hits: lut1",50,0,5);
  TH1F *good_hits_lut2 = new TH1F("good_hits_lut2","good_hits: lut2",50,0,5);
  TH1F *good_hits_lut3 = new TH1F("good_hits_lut3","good_hits: lut3",150,0,15);
  TH1F *good_hits_lut4 = new TH1F("good_hits_lut4","good_hits: lut4",300,5,30);
  TH1F *eff_lut1 = new TH1F("eff_lut1","Efficiency: |ieta| < 21 ",50,0,5);
  TH1F *eff_lut2 = new TH1F("eff_lut2","Efficiency: 21 <= |ieta| < 27",50,0,5);
  TH1F *eff_lut3 = new TH1F("eff_lut3","Efficiency: 27 <= |ieta| < 29",150,0,15);
  TH1F *eff_lut4 = new TH1F("eff_lut4","Efficiency: 29 <= |ieta|",300,5,30);
  TH1F *effsum_lut1= new TH1F("effsum_lut1","Efficiency Summary: |ieta| < 21",50,0,1.1);
  TH1F *effsum_lut2= new TH1F("effsum_lut2","Efficiency Summary: 21 <= |ieta| < 27",50,0,1.1);
  TH1F *effsum_lut3= new TH1F("effsum_lut3","Efficiency Summary: 27 <= |ieta| < 29",50,0,1.1);
  TH1F *effsum_lut4= new TH1F("effsum_lut4","Efficiency Summary: 29 <= |ieta|",50,0,1.1);
  TH1F *threshsum_lut1= new TH1F("threshsum_lut1","Threshold Summary: |ieta| < 21",100,0,2);
  TH1F *threshsum_lut2= new TH1F("threshsum_lut2","Threshold Summary: 21 <= |ieta| < 27",100,0,2);
  TH1F *threshsum_lut3= new TH1F("threshsum_lut3","Threshold Summary: 27 <= |ieta| < 29",500,0,10);
  TH1F *threshsum_lut4= new TH1F("threshsum_lut4","Threshold Summary: 29 <= |ieta|",1000,0,20);
  TH1F *widthsum_lut1= new TH1F("widthsum_lut1","Width Summary: |ieta| < 21",100,0,1);
  TH1F *widthsum_lut2= new TH1F("widthsum_lut2","Width Summary: 21 <= |ieta| < 27",100,0,1);
  TH1F *widthsum_lut3= new TH1F("widthsum_lut3","Width Summary: 27 <= |ieta| < 29",100,0,1);
  TH1F *widthsum_lut4= new TH1F("widthsum_lut4","Width Summary: 29 <= |ieta|",150,0,1.5);
  TH1F *nofiresum_lut1 = new TH1F("nofiresum_lut1","Highest Energy with No TPG Summary: |ieta| < 21", 90,0,30);
  TH1F *nofiresum_lut2 = new TH1F("nofiresum_lut2","Highest Energy with No TPG Summary: 21 <= |ieta| < 27", 90,0,30);
  TH1F *nofiresum_lut3 = new TH1F("nofiresum_lut3","Highest Energy with No TPG Summary: 27 <= |ieta| < 29", 90,0,30);
  TH1F *nofiresum_lut4 = new TH1F("nofiresum_lut4","Highest Energy with No TPG Summary: 29 <= |ieta|", 90,0,30);
  TH1F *fakessum_lut1 = new TH1F("fakessum_lut1","Energy of Fake Hits: |ieta| < 21",90,0,30);
  TH1F *fakessum_lut2 = new TH1F("fakessum_lut2","Energy of Fake Hits: 21 <= |ieta| < 27",90,0,30);
  TH1F *fakessum_lut3 = new TH1F("fakessum_lut3","Energy of Fake Hits: 27 <= |ieta| < 29",90,0,30);
  TH1F *fakessum_lut4 = new TH1F("fakessum_lut4","Energy of Fake Hits: 29 <= |ieta|",90,0,30);


  TObjArray all_hits(4176);
  TObjArray good_hits(4176);
  TObjArray efficiency(4176);
  TObjArray hit_tpg(4176);
  TObjArray resolution(4176);


  char name[20], title[20];
  map<int,float> tpg_energy_map, uncompressed_map, rec_energy_map;
  float compressed, uncompressed, rec;

  for(int i=0;i<= (int)(tree->GetEntries() ) - 1;i++) 
    {
      tree->GetEntry(i);
      tpg_energy_map.clear();
      uncompressed_map.clear();
      rec_energy_map.clear();
      //initialize histos and fill maps/arrays
      for (int j=0; j<4176; ++j)
	{
	  true_index[j] = get_index.ntpg(j);
	  ieta[j] = true_index[j]/100;
	  iphi[j] = TMath::Abs(true_index[j]%100);
	  if (tpg_index[j] != 0)
	    {
	      tpg_energy_map[tpg_index[j]] = tpg_energy[j];
	      rec_energy_map[tpg_index[j]] = rec_energy[j];
	      uncompressed_map[tpg_index[j]] = et2e(TMath::Abs(ieta[j]))*tpg_uncompressed[j];
	      //cout << "channel found: index = " << tpg_index[j] << " tpg counts = " << tpg_energy[j] << "\n";

	    }
	  if (i==0)
	    {
	      sprintf(name,"all_%d",true_index[j]);
	      sprintf(title,"all hits:%d",true_index[j]);
	      all_hits[j] = new TH1F(name,title,200,0,20);
	      sprintf(name,"good_%d",true_index[j]);
	      sprintf(title,"good hits:%d",true_index[j]);
	      good_hits[j] = new TH1F(name,title,200,0,20);
	      sprintf(name,"hit_tpg_%d",true_index[j]);
              sprintf(title,"hit_tpg:%d",true_index[j]);
              hit_tpg[j] = new TProfile(name,title,200,0,100,0,200);
	      sprintf(name,"resolution_%d",true_index[j]);
              sprintf(title,"resolution:%d",true_index[j]);
              resolution[j] = new TH1F(name,title,20,-5,5);
   	    }
	}
      //fill histos with newly filled maps
      //use -1 as the default value for a channel w/o an ide
      for (int j=0; j<4176; j++)
	{
	  if (tpg_energy_map.count(true_index[j])==0)
	    {
	      compressed = -1;
	    }
	  else
	    {
	      compressed = tpg_energy_map.find(true_index[j])->second;
	    }
	  if (uncompressed_map.count(true_index[j])==0)
	    {
	      uncompressed = -1;
	    }
	  else
	    {
	      uncompressed = uncompressed_map.find(true_index[j])->second;
	    }
	  if (rec_energy_map.count(true_index[j])==0)
	    {
	      rec = -1;
	    }
	  else
	    {
	      rec = rec_energy_map.find(true_index[j])->second;
	    }
	  
	  if(compressed != -1 && uncompressed != -1 && rec != -1)
	    {
	      ((TH1F*)all_hits[j])->Fill(rec);
	      if (Abs(ieta[j]) <= 20)
		{
		  all_hits_lut1->Fill(rec);
		  if (compressed > 0)
		    {
		      good_hits_lut1->Fill(rec);
		      hit_tpg_lut1->Fill(rec,uncompressed);
		      res_hit_lut1->Fill(rec,(rec-uncompressed)/rec);
		      ((TH1F*)good_hits[j])->Fill(rec);
		      ((TProfile*)hit_tpg[j])->Fill(rec,uncompressed);
		      ((TH1F*)resolution[j])->Fill((rec-uncompressed)/rec);
		      if (rec == 0)
			{
			  fakes->Fill(ieta[j],iphi[j]);
			  fakessum_lut1->Fill(uncompressed);
			}
		    }
		}
	      else if (Abs(ieta[j]) <= 26)
		{              
		  all_hits_lut2->Fill(rec);
		  if (compressed > 0)
		    {
		      good_hits_lut2->Fill(rec);
		      hit_tpg_lut2->Fill(rec,uncompressed);
		      res_hit_lut2->Fill(rec,(rec-uncompressed)/rec);
		      ((TH1F*)good_hits[j])->Fill(rec);
		      ((TProfile*)hit_tpg[j])->Fill(rec,uncompressed);
		      ((TH1F*)resolution[j])->Fill((rec-uncompressed)/rec);
		      if (rec == 0)
			{
			  fakes->Fill(ieta[j],iphi[j]);
			  fakessum_lut2->Fill(uncompressed);
			}
		    }
		}
	      else if (Abs(ieta[j]) <= 28)
		{
		  all_hits_lut3->Fill(rec);
		  if (compressed > 0)
		    {
		      good_hits_lut3->Fill(rec);
		      hit_tpg_lut3->Fill(rec,uncompressed);
		      res_hit_lut3->Fill(rec,(rec-uncompressed)/rec);
		      ((TH1F*)good_hits[j])->Fill(rec);
		      ((TProfile*)hit_tpg[j])->Fill(rec,uncompressed);
		      ((TH1F*)resolution[j])->Fill((rec-uncompressed)/rec);
		      if (rec == 0)
			{
			  fakes->Fill(ieta[j],iphi[j]);
			  fakessum_lut3->Fill(uncompressed);
			}
		    }
		}
	      else
		{
		  all_hits_lut4->Fill(rec);
		  if (compressed > 0)
		    {
		      good_hits_lut4->Fill(rec);
		      hit_tpg_lut4->Fill(rec,uncompressed);
		      res_hit_lut4->Fill(rec,(rec-uncompressed)/rec);
		      ((TH1F*)good_hits[j])->Fill(rec);
		      ((TProfile*)hit_tpg[j])->Fill(rec,uncompressed);
		      ((TH1F*)resolution[j])->Fill((rec-uncompressed)/rec);
		      if (rec == 0)
			{
			  fakes->Fill(ieta[j],iphi[j]);
			  fakessum_lut4->Fill(uncompressed);
			}
		    }
		}
	      //Fill nofires
	      if (compressed == 0)
		{
		  if (rec > highest_energy[j])
		    {
		      highest_energy[j] = rec;
		    }
		}
	    }
	}
    }
  
  TF1 *fit = new TF1("fit","[0]*x",0,100);
  TF1 *turnon = new TF1("turnon","[0]*0.5*(TMath::Erf((x -[1])*0.5/[2])+1.)",0,30);

  eff_lut1->Divide(good_hits_lut1,all_hits_lut1,1,1,"B");
  eff_lut2->Divide(good_hits_lut2,all_hits_lut2,1,1,"B");
  eff_lut3->Divide(good_hits_lut3,all_hits_lut3,1,1,"B");
  eff_lut4->Divide(good_hits_lut4,all_hits_lut4,1,1,"B");

  turnon->SetParameter(0,1);
  turnon->SetParameter(1,2.5);
  turnon->SetParameter(2,2);
  eff_lut1->Fit("turnon");

  turnon->SetParameter(0,1);
  turnon->SetParameter(1,2.5);
  turnon->SetParameter(2,2);
  eff_lut2->Fit("turnon");

  turnon->SetParameter(0,1);
  turnon->SetParameter(1,7.5);
  turnon->SetParameter(2,3);
  eff_lut3->Fit("turnon");

  turnon->SetParameter(0,1);
  turnon->SetParameter(1,12.5);
  turnon->SetParameter(2,4);
  eff_lut4->Fit("turnon");

  double mean;
  double sigma;
  double thresh;
  double width;
  double effic;

  for (int j=0; j< 4176; ++j)
    {
      turnon->SetParameter(0,1);
      turnon->SetParameter(1,10);
      turnon->SetParameter(2,6);
      //no fire plot
      nofire->Fill(ieta[j],iphi[j],highest_energy[j]);
      
      //efficiency plots
      sprintf(name,"eff%d",true_index[j]);
      sprintf(title,"efficiency:%d",true_index[j]);
      efficiency[j] = new TH1F(name,title,200,0,20);
      ((TH1F*)efficiency[j])->Divide((TH1F*)good_hits[j],(TH1F*)all_hits[j],1,1,"B");
      ((TH1F*)efficiency[j])->Fit("turnon");
      effic = turnon->GetParameter(0);
      thresh = turnon->GetParameter(1);
      width = turnon->GetParameter(2);
      heffic->Fill(ieta[j],iphi[j],effic);
      hthresh->Fill(ieta[j],iphi[j],thresh);
      hwidth->Fill(ieta[j],iphi[j],width);
    
      //resolution plots
      mean = ((TH1F*)resolution[j])->GetMean(1);
      sigma = ((TH1F*)resolution[j])->GetRMS(1);
      res2D_mean->Fill(ieta[j],iphi[j],mean);
      res2D_rms->Fill(ieta[j],iphi[j],sigma);


      if (Abs(ieta[j]) <= 20)
	{
	  effsum_lut1->Fill(effic);
	  threshsum_lut1->Fill(thresh);
	  widthsum_lut1->Fill(width);
	  res1D_mean_lut1->Fill(mean);
	  res1D_rms_lut1->Fill(sigma);
	  nofiresum_lut1->Fill(highest_energy[j]);
	}
      else if (Abs(ieta[j]) <=26)
	{
          effsum_lut2->Fill(effic);
          threshsum_lut2->Fill(thresh);
          widthsum_lut2->Fill(width);
	  res1D_mean_lut2->Fill(mean);
          res1D_rms_lut2->Fill(sigma);
          nofiresum_lut2->Fill(highest_energy[j]);
	}
      else if (Abs(ieta[j]) <= 28)
	{
          effsum_lut3->Fill(effic);
          threshsum_lut3->Fill(thresh);
          widthsum_lut3->Fill(width);
	  res1D_mean_lut3->Fill(mean);
          res1D_rms_lut3->Fill(sigma);
          nofiresum_lut3->Fill(highest_energy[j]);
	}
      else
	{
          effsum_lut4->Fill(effic);
          threshsum_lut4->Fill(thresh);
          widthsum_lut4->Fill(width);
	  res1D_mean_lut4->Fill(mean);
          res1D_rms_lut4->Fill(sigma);
          nofiresum_lut4->Fill(highest_energy[j]);
	}
      
      //lut testing plots
      ((TProfile*)hit_tpg[j])->Fit("fit","","",1,100);
      Double_t slope = fit->GetParameter(0);
      hslope->Fill(ieta[j],iphi[j],slope);
      slope_sum->Fill(slope);
      slope_eta->Fill(TMath::Abs(ieta[j]),slope);
    }
  delete c1;
  

  //efficiency
  //raw efficiencies
  TCanvas *c2 = new TCanvas();
  gStyle->SetOptStat("e");
  c2->Divide(2,2);
  c2->cd(1);
  eff_lut1->Draw();
  SetupTitle(eff_lut1,"Rec Energy (GeV)","Efficiency");
  SetStatus(eff_lut1,"UNCHECKED");
  c2->cd(2);
  eff_lut2->Draw();
  SetupTitle(eff_lut2,"Rec Energy (GeV)","Efficiency");
  SetStatus(eff_lut2,"UNCHECKED");  
  c2->cd(3);
  eff_lut3->Draw();
  SetupTitle(eff_lut3,"Rec Energy (GeV)","Efficiency");
  SetStatus(eff_lut3,"UNCHECKED");
  c2->cd(4);
  eff_lut4->Draw();
  SetupTitle(eff_lut4,"Rec Energy (GeV)","Efficiency");
  SetStatus(eff_lut4,"UNCHECKED");
  c2->Print("output.ps(");
  delete c2;

  //tower by tower efficiencies
  TCanvas *c3 = new TCanvas();
  heffic->Draw("COLZ");
  SetupTowerDisplay(heffic);
  c3->Print("output.ps");
  delete c3;

  TCanvas *c4 = new TCanvas();
  hthresh->Draw("COLZ");
  SetupTowerDisplay(hthresh);
  c4->Print("output.ps");
  delete c4;

  TCanvas *c5 = new TCanvas();
  hwidth->Draw("COLZ");
  SetupTowerDisplay(hwidth);
  c5->Print("output.ps");
  delete c5;

  //efficiency summary
  TCanvas *c6 = new TCanvas();
  gStyle->SetOptStat("emruo");
  c6->Divide(2,2);
  c6->cd(1);
  effsum_lut1->Draw();
  SetupTitle(effsum_lut1,"Efficiency","Ntowers");
  SetStatus(effsum_lut1,"UNCHECKED");
  c6->cd(2);
  effsum_lut2->Draw();
  SetupTitle(effsum_lut2,"Efficiency","Ntowers");
  SetStatus(effsum_lut2,"UNCHECKED");
  c6->cd(3);
  effsum_lut3->Draw();
  SetupTitle(effsum_lut3,"Efficiency","Ntowers");
  SetStatus(effsum_lut3,"UNCHECKED");
  c6->cd(4);
  effsum_lut4->Draw();
  SetupTitle(effsum_lut4,"Efficiency","Ntowers");
  SetStatus(effsum_lut4,"UNCHECKED");
  c6->Print("output.ps");
  delete c6;

  TCanvas *c7 = new TCanvas();
  gStyle->SetOptStat("emruo");
  c7->Divide(2,2);
  c7->cd(1);
  threshsum_lut1->Draw();
  SetupTitle(threshsum_lut1,"Threshold","Ntowers");
  SetStatus(threshsum_lut1,"UNCHECKED");
  c7->cd(2);
  threshsum_lut2->Draw();
  SetupTitle(threshsum_lut2,"Threshold","Ntowers");
  SetStatus(threshsum_lut2,"UNCHECKED");
  c7->cd(3);
  threshsum_lut3->Draw();
  SetupTitle(threshsum_lut3,"Threshold","Ntowers");
  SetStatus(threshsum_lut3,"UNCHECKED");
  c7->cd(4);
  threshsum_lut4->Draw();
  SetupTitle(threshsum_lut4,"Threshold","Ntowers");
  SetStatus(threshsum_lut4,"UNCHECKED");
  c7->Print("output.ps");
  delete c7;

  TCanvas *c8 = new TCanvas();
  c8->Divide(2,2);
  gStyle->SetOptStat("emruo");
  c8->cd(1);
  widthsum_lut1->Draw();
  SetupTitle(widthsum_lut1,"Width","Ntowers");
  SetStatus(widthsum_lut1,"UNCHECKED");
  c8->cd(2);
  widthsum_lut2->Draw();
  SetupTitle(widthsum_lut2,"Width","Ntowers");
  SetStatus(widthsum_lut2,"UNCHECKED");
  c8->cd(3);
  widthsum_lut3->Draw();
  SetupTitle(widthsum_lut3,"Width","Ntowers");
  SetStatus(widthsum_lut3,"UNCHECKED");
  c8->cd(4);
  widthsum_lut4->Draw();
  SetupTitle(widthsum_lut4,"Width","Ntowers");
  SetStatus(widthsum_lut4,"UNCHECKED");
  c8->Print("output.ps");
  delete c8;

  //slopes
  //raw slope
  TCanvas *c15 = new TCanvas();
  gStyle->SetOptStat("e");
  c15->Divide(2,2);
  TF1 *line = new TF1("line","x",0,100);
  line->SetLineColor(4);
  c15->cd(1);
  hit_tpg_lut1->Draw("box");
  line->Draw("same");
  SetupTitle(hit_tpg_lut1,"Rec Energy (Gev)","Uncompressed TPG (GeV)");
  c15->cd(2);
  hit_tpg_lut2->Draw("box");
  line->Draw("same");
  SetupTitle(hit_tpg_lut2,"Rec Energy (Gev)","Uncompressed TPG (GeV)");
  c15->cd(3);
  hit_tpg_lut3->Draw("box");
  line->Draw("same");
  SetupTitle(hit_tpg_lut3,"Rec Energy (Gev)","Uncompressed TPG (GeV)");
  c15->cd(4);
  hit_tpg_lut4->Draw("box");
  line->Draw("same");
  SetupTitle(hit_tpg_lut4,"Rec Energy (Gev)","Uncompressed TPG (GeV)");
  c15->Print("output.ps");
  delete c15;

  //tower by tower slope
  TCanvas *c16 = new TCanvas();
  hslope->Draw("COLZ");
  SetupTowerDisplay(hslope);
  c16->Print("output.ps");
  delete c16;

  //slope summary
  TCanvas *c17 = new TCanvas();
  gStyle->SetOptStat("emruo");
  slope_sum->Draw();
  SetupTitle(slope_sum,"Slope of Uncompressed TPG vs Rec Energy","Ntowers");
  SetStatus(slope_sum,"UNCHECKED");
  c17->SetLogy();
  c17->Print("output.ps");
  delete c17;

  //resolution
  //raw resoltions
  TCanvas *c9 = new TCanvas();
  gStyle->SetOptStat("e");
  c9->Divide(2,2);
  c9->cd(1);
  res_hit_lut1->Draw("box");
  SetupTitle(res_hit_lut1,"Rec. Hit Energy (GeV)","(Rec E - Uncompressed TPG)/Rec E");
  c9->cd(2);
  res_hit_lut2->Draw("box");
  SetupTitle(res_hit_lut2,"Rec. Hit Energy (GeV)","(Rec E - Uncompressed TPG)/Rec E");
  c9->cd(3);
  res_hit_lut3->Draw("box");
  SetupTitle(res_hit_lut3,"Rec. Hit Energy (GeV)","(Rec E - Uncompressed TPG)/Rec E");
  c9->cd(4);
  res_hit_lut4->Draw("box");
  SetupTitle(res_hit_lut4,"Rec. Hit Energy (GeV)","(Rec E - Uncompressed TPG)/Rec E");
  c9->Print("output.ps");
  delete c9;

  //tower by tower resolutions
  TCanvas *c10 = new TCanvas();
  res2D_mean->Draw("COLZ");
  SetupTowerDisplay(res2D_mean);
  c10->Print("output.ps");
  delete c10;

  TCanvas *c11 = new TCanvas();
  res2D_rms->Draw("COLZ");
  SetupTowerDisplay(res2D_rms);
  c11->Print("output.ps");
  delete c11;
 
  TCanvas *c13 = new TCanvas();
  gStyle->SetOptStat("emruo");
  c13->Divide(2,2);
  c13->cd(1);
  res1D_mean_lut1->Draw();
  SetupTitle(res1D_mean_lut1,"Mean (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_mean_lut1,"UNCHECKED");
  c13->cd(2);
  res1D_mean_lut2->Draw();
  SetupTitle(res1D_mean_lut2,"Mean (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_mean_lut2,"UNCHECKED");
  c13->cd(3);
  res1D_mean_lut3->Draw();
  SetupTitle(res1D_mean_lut3,"Mean (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_mean_lut3,"UNCHECKED");
  c13->cd(4);
  res1D_mean_lut4->Draw();
  SetupTitle(res1D_mean_lut4,"Mean (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_mean_lut4,"UNCHECKED");
  c13->Print("output.ps");
  delete c13;

  TCanvas *c14 = new TCanvas();
  c14->Divide(2,2);
  c14->cd(1);
  res1D_rms_lut1->Draw();
  SetupTitle(res1D_rms_lut1,"RMS (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_rms_lut1,"UNCHECKED");
  c14->cd(2);
  res1D_rms_lut2->Draw();
  SetupTitle(res1D_rms_lut2,"RMS (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_rms_lut2,"UNCHECKED");
  c14->cd(3);
  res1D_rms_lut3->Draw();
  SetupTitle(res1D_rms_lut3,"RMS (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_rms_lut3,"UNCHECKED");
  c14->cd(4);
  res1D_rms_lut4->Draw();
  SetupTitle(res1D_rms_lut4,"RMS (Rec E - Uncompressed TPG)/Rec E","Ntowers");
  SetStatus(res1D_rms_lut4,"UNCHECKED");
  c14->Print("output.ps");
  delete c14;

  //nofire
  TCanvas *c18 = new TCanvas();
  nofire->Draw("COLZ");
  SetupTowerDisplay(nofire);
  c18->Print("output.ps");
  delete c18;

  //nofiresum
  TCanvas *c20 = new TCanvas();
  gStyle->SetOptStat("emruo");
  c20->Divide(2,2);
  c20->cd(1);
  nofiresum_lut1->Draw();
  SetupTitle(nofiresum_lut1,"Highest Energy with no TPG (GeV)","Ntowers");
  SetStatus(nofiresum_lut1,"UNCHECKED");
  c20->cd(2);
  SetupTitle(nofiresum_lut2,"Highest Energy with no TPG (GeV)","Ntowers");
  SetStatus(nofiresum_lut2,"UNCHECKED");
  nofiresum_lut2->Draw();
  c20->cd(3);
  SetupTitle(nofiresum_lut3,"Highest Energy with no TPG (GeV)","Ntowers");
  SetStatus(nofiresum_lut3,"UNCHECKED");
  nofiresum_lut3->Draw();
  c20->cd(4);
  SetupTitle(nofiresum_lut4,"Highest Energy with no TPG (GeV)","Ntowers");
  SetStatus(nofiresum_lut4,"UNCHECKED");
  nofiresum_lut4->Draw();
  c20->Print("output.ps");
  delete c20;

  //fakes
  TCanvas *c19 = new TCanvas();
  fakes->Draw("COLZ");
  SetupTowerDisplay(fakes);
  c19->Print("output.ps");
  delete c19;

  TCanvas *c21 = new TCanvas();
  gStyle->SetOptStat("emruo");
  c21->Divide(2,2);
  c21->cd(1);
  fakessum_lut1->Draw();
  SetupTitle(fakessum_lut1,"Uncompressed Energy of Fake TP (GeV)","Ntowers");
  SetStatus(fakessum_lut1,"UNCHECKED");
  c21->cd(2);
  fakessum_lut2->Draw();
  SetupTitle(fakessum_lut2,"Uncompressed Energy of Fake TP (GeV)","Ntowers");
  SetStatus(fakessum_lut2,"UNCHECKED");
  c21->cd(3);
  fakessum_lut3->Draw();
  SetupTitle(fakessum_lut3,"Uncompressed Energy of Fake TP (GeV)","Ntowers");
  SetStatus(fakessum_lut3,"UNCHECKED");
  c21->cd(4);
  fakessum_lut4->Draw();
  SetupTitle(fakessum_lut4,"Uncompressed Energy of Fake TP (GeV)","Ntowers");
  SetStatus(fakessum_lut4,"UNCHECKED");
  c21->Print("output.ps)");
  delete c21;

  TFile f(output_file,"recreate");
  efficiency.Write();
  nofire->Write();
  fakes->Write();
  resolution.Write();
  hit_tpg.Write();
  hslope->Write();
  slope_sum->Write();
  slope_eta->Write();
  res2D_mean->Write();
  res2D_rms->Write();
  res1D_mean_lut1->Write();
  res1D_rms_lut1->Write();
  res1D_mean_lut2->Write();
  res1D_rms_lut2->Write(); 
  res1D_mean_lut3->Write();
  res1D_rms_lut3->Write(); 
  res1D_mean_lut4->Write();
  res1D_rms_lut4->Write();
  heffic->Write();
  hthresh->Write();
  hwidth->Write();
  eff_lut1->Write();
  eff_lut2->Write();
  eff_lut3->Write();
  eff_lut4->Write();
  hit_tpg_lut1->Write();
  hit_tpg_lut2->Write();
  hit_tpg_lut3->Write();
  hit_tpg_lut4->Write();
  res_hit_lut1->Write();
  res_hit_lut2->Write();
  res_hit_lut3->Write();
  res_hit_lut4->Write();
  effsum_lut1->Write();
  effsum_lut2->Write();
  effsum_lut3->Write();
  effsum_lut4->Write();
  threshsum_lut1->Write();
  threshsum_lut2->Write();
  threshsum_lut3->Write();
  threshsum_lut4->Write();
  widthsum_lut1->Write();
  widthsum_lut2->Write();
  widthsum_lut3->Write();
  widthsum_lut4->Write();
  fakessum_lut1->Write();
  fakessum_lut2->Write();
  fakessum_lut3->Write();
  fakessum_lut4->Write();
  nofiresum_lut1->Write();
  nofiresum_lut2->Write();
  nofiresum_lut3->Write();
  nofiresum_lut4->Write();
  f.Close();
  
  file->Close();
  return 0;
}

double et2e(int eta)
{
  switch (TMath::Abs(eta))
    {
    case 1:	 
      return 1.001;
    case 2:
      return 1.009;
    case 3:
      return 1.024;
    case 4:
      return 1.047;
    case 5:
      return 1.078;
    case 6:
      return 1.117;
    case 7:
      return 1.164;
    case 8:
      return 1.221;
    case 9:
      return 1.286;
    case 10:
      return 1.361;
    case 11:	
      return 1.447;
    case 12:
      return 1.544;
    case 13:	 
      return 1.652;
    case 14:	 
      return 1.773;
    case 15:
      return 1.907;
    case 16:
      return 2.056;
    case 17:
      return 2.220;
    case 18:
      return 2.401;
    case 19:	 
      return 2.600;
    case 20:	 
      return 2.819;
    case 21:
      return 3.064;
    case 22:	
      return 3.353;
    case 23:	 
      return 3.714;
    case 24:
      return 4.175;
    case 25:	
      return 4.806;
    case 26:	
      return 5.645;
    case 27:	 
      return 6.604;
    case 28:	 
      return 8.460;
    case 29:
      return 10.94;
    case 30:
      return 17.89;
    case 31:	 
      return 30.21;
    case 32:
      return 59.38;
    default: 
      return -99999;
    }
}
