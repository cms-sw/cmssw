/*

Plotter to plot various histograms produced by benchmarking.C

Usage : root -l 
        .L plot.C
        plot()

Input : track_validation_histos.root (produced after running benchmarking)

Options : plot(0) --> RoadSearch Histograms only
          plot(1) --> CTF Histograms only
          plot(2) --> Both Histograms (Comparison)

Original Authors : T. Moulik (tmoulik@fnal.gov, Tania.Moulik@cern.ch)
                   C. Hinchey (chinchey@ku.edu)

*/

void plot(int draw=2) {
  //draw  0 = RS, 1 = CTF, 2=Both

  char *myfile = "track_validation_histos.root";
  TFile *file = new TFile(myfile);
  gStyle->SetOptStat(1110);
  gStyle->SetStatH(0.30);
  gStyle->SetStatW(0.40);
  gStyle->SetStatFormat("5.3g");
  gStyle->SetLabelSize(0.04);
  


  unsigned int ipt = 5; //array number of plots to use

  TString ptname = Form("pttr_pt_%d",ipt);
  TString ptname_ctf = Form("pttr_pt_%d_ctf",ipt);
  TString d0name = Form("pttr_d0_%d",ipt);
  TString d0name_ctf = Form("pttr_d0_%d_ctf",ipt);
  TString z0name = Form("pttr_z0_%d",ipt);
  TString z0name_ctf = Form("pttr_z0_%d_ctf",ipt);
  TString etares_name = Form("pttr_etares_%d",ipt);
  TString etares_name_ctf = Form("pttr_etares_%d_ctf");
  TString phires_name = Form("pttr_phires_%d",ipt);
  TString phires_name_ctf = Form("pttr_phires_%d_ctf");
  TString geneta_name = "pttr_geneta";
  TString genphi_name = "pttr_genphi";
  TString recoeta_name     = Form("pttr_eta_%d",ipt);
  TString recoeta_name_ctf = Form("pttr_eta_%d_ctf",ipt);
  TString recophi_name     = Form("pttr_phi_%d",ipt);     
  TString recophi_name_ctf = Form("pttr_phi_%d_ctf",ipt);

  TH1F *h1[7];
  TH1F *h1_ctf[7];
  
  h1[0] = (TH1F*)gROOT->FindObject(ptname);
  h1_ctf[0] = (TH1F*)gROOT->FindObject(ptname_ctf);
  h1[1] = (TH1F*)gROOT->FindObject(d0name);
  h1_ctf[1] = (TH1F*)gROOT->FindObject(d0name_ctf);
  h1[2] = (TH1F*)gROOT->FindObject(z0name);
  h1_ctf[2] = (TH1F*)gROOT->FindObject(z0name_ctf);
  h1[3] = (TH1F*)gROOT->FindObject(recoeta_name);
  h1_ctf[3] = (TH1F*)gROOT->FindObject(recoeta_name_ctf);
  h1[4] = (TH1F*)gROOT->FindObject(recophi_name);
  h1_ctf[4] = (TH1F*)gROOT->FindObject(recophi_name_ctf);
  h1[5] = (TH1F*)gROOT->FindObject(etares_name);
  h1_ctf[5] = (TH1F*)gROOT->FindObject(etares_name_ctf);
  h1[6] = (TH1F*)gROOT->FindObject(phires_name);
  h1_ctf[6] = (TH1F*)gROOT->FindObject(phires_name_ctf);

  TH1F *gen_eta = (TH1F*)gROOT->FindObject(geneta_name);
  TH1F *gen_phi = (TH1F*)gROOT->FindObject(genphi_name);


  TCanvas *c1 = new TCanvas("plot_1","plot_1",700,700);
  c1->Divide(2,2);
  gROOT->ProcessLine(".L histoper.C+");
  histoper op;

  for (int i=0; i<3; i++) { // plot 1
    cout << i << endl;
    c1->cd(i+1);
    h1[i]->SetLineColor(4);
    h1_ctf[i]->SetLineColor(2);
    if (draw==0) h1[i]->Draw();
    if (draw==1) h1_ctf[i]->Draw();
    if (draw==2) {
      if(h1[i]->GetMaximum() > h1_ctf[i]->GetMaximum() ) {
	h1[i]->Draw();
	h1_ctf[i]->Draw("SAME");
      }
      else {
	h1_ctf[i]->Draw();
	h1[i]->Draw("SAME");
      }
      Double_t x1=0.1;
      Double_t y1=0.7;
      Double_t dx = 0.25;
      Double_t dy = 0.2;
      Double_t x2=x1+dx;
      Double_t y2=y1+dy;
      leg = new TLegend(x1,y1,x2,y2);
      leg->AddEntry(h1[i],"RSKF","l");
      leg->AddEntry(h1_ctf[i],"CTFKF","l");
      leg->Draw();  
    }
  }

  c1->cd();


  TCanvas *c2 = new TCanvas("plot_2","plot_2",700,350);
  c2->Divide(2,1);

  // plot 2
  TFormula *f1 = new TFormula("fitgaus", "gaus(0)");
  TF1 *f2 = new TF1("f2","fitgaus");
  float ymax = h1[5]->GetMaximum();
  float mean = h1[5]->GetMean();
  float rms  = h1[5]->GetRMS();
  f2->SetParameters(ymax/3.0,mean,rms);
  f2->SetParLimits(0,0.0,0.0);
  h1[5]->Fit("f2","0");
  
  ymax = h1_ctf[5]->GetMaximum();
  mean = h1_ctf[5]->GetMean();
  rms  = h1_ctf[5]->GetRMS();
  f2->SetParameters(ymax/3.0,mean,rms);
  f2->SetParLimits(0,0.0,0.0);
  h1_ctf[5]->Fit("f2","0");
  
  c2->cd(1);
  h1[3]->SetTitle("efficiency vs eta");
  h1_ctf[3]->SetTitle("efficiency vs eta");
  TH1F *h13 = op.div(h1[3],gen_eta,"eff3");
  TH1F* h1_ctf3 = op.div(h1_ctf[3],gen_eta,"eff3ctf");
  h13->SetLineColor(4);
  h1[5]->SetLineColor(4);
  h1_ctf3->SetLineColor(2);
  h1_ctf[5]->SetLineColor(2);
  if (draw==0){
    h13->Draw();
    c2->cd(2);
    h1[5]->Draw();
  }
  if (draw==1){
    h1ctf3->Draw();
    c2->cd(2);
    h1_ctf[5]->Draw();
  }
  if (draw==2) {
    if(h13->GetMaximum() > h1_ctf3->GetMaximum() ) {
      h13->Draw();
      h1_ctf3->Draw("SAME");
    }
    else {
      h1_ctf3->Draw();
      h13->Draw("SAME");
    }

    Double_t x1=0.1;
    Double_t y1=0.7;
    Double_t dx = 0.25;
    Double_t dy = 0.2;
    Double_t x2=x1+dx;
    Double_t y2=y1+dy;
    leg = new TLegend(x1,y1,x2,y2);
    leg->AddEntry(h1[3],"RSKF","l");
    leg->AddEntry(h1_ctf[3],"CTFKF","l");
    leg->Draw();  
    
    c2->cd(2);
    if(h1[5]->GetMaximum() > h1_ctf[5]->GetMaximum() ) {
      h1[5]->Draw();
      h1_ctf[5]->Draw("SAME");
    }
    else {
      h1_ctf[5]->Draw();
      h1[5]->Draw("SAME");
    }
    
    leg2 = new TLegend(x1,y1,x2,y2);
    leg2->AddEntry(h1[5],"RSKF","l");
    leg2->AddEntry(h1_ctf[5],"CTFKF","l");
    leg2->Draw();
    
  }


  c2->cd();


  cout << "Opening plot_3" << endl;

  TCanvas *c3 = new TCanvas("plot_3","plot_3",700,350);
  c3->Divide(2,1);
  
  // plot 3
  ymax = h1[6]->GetMaximum();
  mean = h1[6]->GetMean();
  rms  = h1[6]->GetRMS();
  f2->SetParameters(ymax/3.0,mean,rms);
  f2->SetParLimits(0,0.0,0.0);
  h1[6]->Fit("f2","0");
  
  ymax = h1_ctf[6]->GetMaximum();
  mean = h1_ctf[6]->GetMean();
  rms  = h1_ctf[6]->GetRMS();
  f2->SetParameters(ymax/3.0,mean,rms);
  f2->SetParLimits(0,0.0,0.0);
  h1_ctf[6]->Fit("f2","0");
  
  c3->cd(1);
  h1[4]->SetTitle("efficiency vs phi");
  h1_ctf[4]->SetTitle("efficiency vs phi");
  h1[4]->Divide(gen_phi);
  h1_ctf[4]->Divide(gen_phi);
  h1[4]->SetLineColor(4);
  h1[6]->SetLineColor(4);
  h1_ctf[4]->SetLineColor(2);
  h1_ctf[6]->SetLineColor(2);
  if (draw==0){
    h1[4]->Draw();
    c2->cd(2);
    h1[6]->Draw();
  }
  if (draw==1){
    h1_ctf[4]->Draw();
    c2->cd(2);
    h1_ctf[6]->Draw();
  }

  if (draw==2) {
    if(h1[4]->GetMaximum() > h1_ctf[4]->GetMaximum() ) {
      h1[4]->Draw();
      h1_ctf[4]->Draw("SAME");
    }
    else {
      h1_ctf[4]->Draw();
      h1[4]->Draw("SAME");
    }
    Double_t x1=0.1;
    Double_t y1=0.7;
    Double_t dx = 0.25;
    Double_t dy = 0.2;
    Double_t x2=x1+dx;
    Double_t y2=y1+dy;
    leg = new TLegend(x1,y1,x2,y2);
    leg->AddEntry(h1[4],"RSKF","l");
    leg->AddEntry(h1_ctf[4],"CTFKF","l");
    leg->Draw();  
    
    c3->cd(2);
    if(h1[6]->GetMaximum() > h1_ctf[6]->GetMaximum() ) {
      h1[6]->Draw();
      h1_ctf[6]->Draw("SAME");
    }
    else {
      h1_ctf[6]->Draw();
      h1[6]->Draw("SAME");
    }
    
    leg2 = new TLegend(x1,y1,x2,y2);
    leg2->AddEntry(h1[6],"RSKF","l");
    leg2->AddEntry(h1_ctf[6],"CTFKF","l");
    leg2->Draw();
    
  }


  c3->cd();

  TString ptres_name = "ptresvspt";
  TString ptres_name_ctf = "ptresvspt_ctf";
  TString phires_name = "phiresvspt";
  TString phires_name_ctf = "phiresvspt_ctf";
  TString etares_name = "etaresvspt";
  TString etares_name_ctf = "etaresvspt_ctf";

  TGraphErrors *hres[3];
  TGraphErrors *hres_ctf[3];

  hres[0] = (TGraphErrors*)gROOT->FindObject(ptres_name);
  hres[1] = (TGraphErrors*)gROOT->FindObject(phires_name);
  hres[2] = (TGraphErrors*)gROOT->FindObject(etares_name);
  
  hres_ctf[0] = (TGraphErrors*)gROOT->FindObject(ptres_name_ctf);
  hres_ctf[1] = (TGraphErrors*)gROOT->FindObject(phires_name_ctf);
  hres_ctf[2] = (TGraphErrors*)gROOT->FindObject(etares_name_ctf);



  cout << "Opening plot_4" << endl;

  // plot 4
  
  TCanvas* c4 = new TCanvas("plot_4","plot_4",350,350);
  hres[0]->SetLineColor(4);
  hres_ctf[0]->SetLineColor(2);
  if (draw==0) hres[0]->Draw("AP");
  if (draw==1) hres_ctf[0]->Draw("AP");
  if (draw==2) {
    if(hres[0]->GetYaxis()->GetXmax() > hres_ctf[0]->GetYaxis()->GetXmax() ) {
      hres[0]->Draw("AP");
      hres_ctf[0]->Draw("P");
    }
    else {
      hres_ctf[0]->Draw("AP");
      hres[0]->Draw("P");
    }
    Double_t x1=0.1;
    Double_t y1=0.7;
    Double_t dx = 0.25;
    Double_t dy = 0.2;
    Double_t x2=x1+dx;
    Double_t y2=y1+dy;
    leg = new TLegend(x1,y1,x2,y2);
    leg->AddEntry(hres[0],"RSKF","l");
    leg->AddEntry(hres_ctf[0],"CTFKF","l");
    leg->Draw();
  }
  
  
  
  cout << "Opening plot 5 " << endl;

  TCanvas* c5 = new TCanvas("plot_5","plot_5",700,350);
  c5->Divide(2,1);

  for (int i=1; i<3; i++) { // plot 5
    cout << i << endl;
    c5->cd(i);
    hres[i]->SetLineColor(4);
    hres_ctf[i]->SetLineColor(2);
    if (draw==0) hres[i]->Draw("AP");
    if (draw==1) hres_ctf[i]->Draw("AP");
    if (draw==2) {
      if(hres[i]->GetYaxis()->GetXmax() > hres_ctf[i]->GetYaxis()->GetXmax() ) {
	hres[i]->Draw("AP");
	hres_ctf[i]->Draw("SAME");
      }
      else {
	hres_ctf[i]->Draw("AP");
	hres[i]->Draw("SAME");
      }
      Double_t x1=0.1;
      Double_t y1=0.7;
      Double_t dx = 0.25;
      Double_t dy = 0.2;
      Double_t x2=x1+dx;
      Double_t y2=y1+dy;
      leg = new TLegend(x1,y1,x2,y2);
      leg->AddEntry(hres[i],"RSKF","l");
      leg->AddEntry(hres_ctf[i],"CTFKF","l");
      leg->Draw();  
    }
  }  
}


