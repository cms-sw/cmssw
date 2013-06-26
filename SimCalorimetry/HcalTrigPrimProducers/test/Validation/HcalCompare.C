#include "HcalCompare.h"

void compare(char* hardware_file, char* software_file)
{
  TStyle *mystyle = new TStyle("mystyle","my style");
  initStyle(mystyle);
  mystyle->cd();
  index_map::index_map get_index;
  TCanvas *c1 = new TCanvas("c1","comparison");
  TFile *file_hard = TFile::Open(hardware_file);
  if(!file_hard)
    {
      cout << "No hardware file found!";
      return;
    }
  TFile *file_soft = TFile::Open(software_file);
  if(!file_soft)
    {
      cout << "No software file found!";
      return;
    }

  TTree* tree_hard = (TTree*)file_hard->Get("TPGntuple");
  if(!tree_hard)
    {
      cout << "No tree found for hardware!";
      return;
    }

  TTree* tree_soft = (TTree*)file_soft->Get("TPGntuple");
  if(!tree_soft)
    {
      cout << "No tree found for software!";
      return;
    }

  float tpg_energy_hard[4176], tpg_energy_soft[4176];
  int ieta_hard[4176], iphi_hard[4176], ieta_soft[4176], iphi_soft[4176], index_hard[4176], index_soft[4176];

  tree_hard->SetBranchAddress("tpg_energy",tpg_energy_hard);
  tree_hard->SetBranchAddress("ieta",ieta_hard);
  tree_hard->SetBranchAddress("iphi",iphi_hard);
  tree_hard->SetBranchAddress("tpg_index",index_hard);

  tree_soft->SetBranchAddress("tpg_energy",tpg_energy_soft);
  tree_soft->SetBranchAddress("ieta",ieta_soft);
  tree_soft->SetBranchAddress("iphi",iphi_soft);
  tree_soft->SetBranchAddress("tpg_index",index_soft);
  
  TH1F *comparison = new TH1F("comparison","tpg_hardware - tpg_software",400,-200,200);
  TH2F *mean_2d = new TH2F("res2D_mean","Mean of TP_{hardware}-TP_{software}",65,-32,33,72,1,73);
  TH2F *rms_2d = new TH2F("res2D_rms","RMS of TP_{hardware}-TP_{software}",65,-32,33,72,1,73);


  TH1F *hardware = new TH1F("hardware","hardware",200,0,200);
  TH1F *software = new TH1F("software","software",200,0,200);

  TObjArray hard_minus_soft(4176);
  TObjArray hard_v_soft(4176);

  char name[20], title[20];


  int true_index[4176];
  int limit;
  //  cout << "^^^^^\nNumber of Entries: hardware = " << tree_hard->GetEntries() << " software = " << tree_soft->GetEntries() << "\n";
  if ((int)tree_hard->GetEntries() <= (int)tree_soft->GetEntries())
    {
      limit = (int)tree_hard->GetEntries();
    }
  else
    {
      limit = (int)tree_soft->GetEntries();
    }

  map<int,float> tpg_hard_map, tpg_soft_map;  
  for(int i=0; i < limit; i++)
    {
      tree_hard->GetEntry(i);
      tree_soft->GetEntry(i);
      //cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      tpg_hard_map.clear();
      tpg_soft_map.clear();
      for(int j=0; j < 4176; j++)
	{
	  true_index[j] = get_index.ntpg(j);
	  if (index_hard[j] != 0)
	    {
	      tpg_hard_map[index_hard[j]] = tpg_energy_hard[j];
	    }
	  if (index_soft[j] != 0)
	    {
	      tpg_soft_map[index_soft[j]] = tpg_energy_soft[j];
	    }
	  //initialize
	  if (i==0)
	    {
	      sprintf(name,"difference_%d",true_index[j]);
	      sprintf(title,"difference:%d",true_index[j]);
	      hard_minus_soft[j] = new TH1F(name,title,160,-80,80);
	      sprintf(name,"hard_v_soft_%d",true_index[j]);
	      sprintf(title,"hard_v_soft:%d",true_index[j]);
	      hard_v_soft[j] = new TH2F(name,title,200,0,200,200,0,200);
	    }
	}

      for(int j=0; j < 4176; j++)
	{
	  int tpg_hard, tpg_soft;
	  if (tpg_hard_map.count(true_index[j])==0)
	    {
	      tpg_hard = -1;
	    }
	  else
	    {
	      tpg_hard = (int)tpg_hard_map.find(true_index[j])->second;
	    }
          if (tpg_soft_map.count(true_index[j])==0)
            {
              tpg_soft = -1;
            }
          else
            {
              tpg_soft = (int)tpg_soft_map.find(true_index[j])->second;
            }
	  //fill histos
	  if(tpg_hard != -1 && tpg_soft != -1)
	    {
	      ((TH2F*)hard_v_soft[j])->Fill(tpg_soft,tpg_hard);
	      hardware->Fill(tpg_hard);
	      software->Fill(tpg_soft);
	      comparison->Fill(tpg_hard-tpg_soft);
	      ((TH1F*)hard_minus_soft[j])->Fill(tpg_hard-tpg_soft);
	    }
	}
	  
      int hard_count = 0;
      int soft_count = 0;
      //cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
      for(int j=0; j < 4176; j++)
	{
	  if (tpg_energy_hard[j] != 0 || (tpg_energy_hard[j] == 0 && index_hard[j] !=0))
	    {
	      hard_count++;
	      cout << "Found hardware channel: " << index_hard[j] << " tpg counts = " << tpg_energy_hard[j] << "\n";
	    }
	  if (tpg_energy_soft[j] !=0  || (tpg_energy_soft[j] == 0 && index_soft[j] !=0))
	    {
	      soft_count++;
	      cout << "Found software channel: " << index_soft[j] << " tpg counts = " << tpg_energy_soft[j] << "\n";
	    }
	}
      cout << "N hardware channels = " << hard_count << " N software channels = " << soft_count << "\n";

    }

  double mean;
  double sigma;
  TF1 *line = new TF1("line","x",0,200);
  line->SetLineColor(4);
  int eta;
  int phi;
  for (int j=0; j<4176; ++j)
    {
      mean = ((TH1F*)hard_minus_soft[j])->GetMean(1);
      sigma = ((TH1F*)hard_minus_soft[j])->GetRMS(1);
      eta = true_index[j]/100;
      phi = TMath::Abs(true_index[j]%100);
      mean_2d->Fill(eta,phi,mean);
      rms_2d->Fill(eta,phi,sigma);
      ((TH2F*)hard_v_soft[j])->Draw();
      line->Draw("same");
    }
  
  c1->SetLogy();
  comparison->Draw();
  SetupTitle(comparison,"Hardware TP - Software TP (counts)", "Ntowers");
  if (comparison->GetMean(1) == 0 && comparison->GetRMS(1) == 0)
    {
      SetStatus(comparison,"GOOD");
    }
  else
    {
      SetStatus(comparison,"BAD");
    }
  c1->Print("compare_output.ps(");

  TCanvas *c2 = new TCanvas("c2","Mean");
  c2->cd();
  //  mean_2d->SetStats(kFALSE);
  mean_2d->Draw("COLZ");
  SetupTowerDisplay(mean_2d);
  c2->Print("compare_output.ps");

  TCanvas *c3 = new TCanvas("c3","RMS");
  c3->cd();
  //rms_2d->SetStats(kFALSE);
  rms_2d->Draw("COLZ");
  SetupTowerDisplay(rms_2d);
  c3->Print("compare_output.ps");
    
  TCanvas *c4 = new TCanvas("c4","hard and soft");
  c4->Divide(1,2);
  c4->cd(1);
  gPad->SetLogy();
  hardware->Draw();
  c4->cd(2);
  gPad->SetLogy();
  software->Draw();
  c4->Print("compare_output.ps)");
  
  TFile *f = new TFile("output.root","recreate");
  hard_v_soft.Write();
  hard_minus_soft.Write();
  f->Close();
  
  return;
}
