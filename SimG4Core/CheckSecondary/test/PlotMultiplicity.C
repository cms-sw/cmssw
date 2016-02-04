#include <iostream>
#include <vector>
#include <string>
#include <map>

int mstyle[15] = { 29, 29, 20, 21, 20, 21, 22, 21, 22, 20, 21, 22, 20, 20, 21};
int mcolor[15] = {  1,  1,  3,  3,  4,  4,  4,  7,  7,  7,  2,  2,  2, 28, 28};
float msiz[15] = {1.4,1.4,1.1,1.1,1.1,1.1,1.4,1.1,1.4,1.1,1.1,1.4,1.1,1.1,1.1};
int lcolor[15] = {  1,  1,  3,  3,  4,  4,  4,  7,  7,  7,  2,  2,  2, 28, 28};
int lstyle[15] = {  1,  1,  1,  2,  1,  2,  3,  2,  3,  1,  3,  2,  1,  1,  2};
int lwidth[15] = {  1,  1,  1,  2,  1,  2,  2,  2,  2,  1,  2,  2,  1,  1,  2};

void plotMomentum(char target[6], char list[20], char ene[6], char part[4],
		  char dir[12]="histo", char g4ver[20]="G4.9.1.p01") {
  
  setStyle();
  gStyle->SetOptLogy(1);

  std::vector<std::string> types = typesOld();
  int energy = atoi(ene);
  
  char ofile[100];
  sprintf (ofile, "%s/histo_%s%s_%s_%sGeV.root", dir, target, list, part, ene);
  std::cout << "Input file " << ofile << "\n";
  TFile *fout = TFile::Open(ofile);
  fout->cd();

  char name[160], title[160], ctype[20], ytitle[20], cname[160];
  TH1F *hiParticle[5][20];
  for (unsigned int ii=0; ii<=(types.size()); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", types[ii-1].c_str());
    for (unsigned int jj=0; jj<5; jj++) {
      sprintf (name, "Particle%i_KE%s%s%sGeV(%s)",jj,target, list, ene, ctype);
      hiParticle[jj][ii] = (TH1F*)fout->FindObjectAny(name);
      hiParticle[jj][ii]->SetFillColor(30);
    }
  }

  TCanvas *c[5];
  for (unsigned int jj=0; jj<2; jj++) {
    hiParticle[jj][11]->Rebin(25);
    hiParticle[jj][12]->Rebin(25);
    hiParticle[jj][13]->Rebin(25);

    sprintf(cname, "c_%s%s%sGeV_nucleons_particle%i", target, list, ene, jj);
    c[jj] = new TCanvas(cname, cname, 800, 800);
    c[jj]->Divide(2,2);
    c[jj]->cd(1); hiParticle[jj][11]->Draw();
    c[jj]->cd(2); hiParticle[jj][12]->Draw();
    c[jj]->cd(3); hiParticle[jj][13]->Draw();
    c[jj]->cd(4); hiParticle[jj][14]->Draw();
  }
}

void plotParticles(char target[6], char list[20], char ene[6], char part[4],
		   char dir[12]="histo", char g4ver[20]="G4.9.1.p01") {
  
  gStyle->SetOptLogy(1);
  gStyle->SetTitleX(.1);
  gStyle->SetTitleY(.9);

  
  std::vector<std::string> types = types();
  char ofile[100];
  sprintf (ofile, "%s/histo_%s%s_%s_%sGeV.root", dir, target, list, part, ene);
  std::cout << "Input file " << ofile << "\n";
  TFile *fout = TFile::Open(ofile);
  fout->cd();

  char ctype[20], title[160], cname[160];
  sprintf(ctype,"Ions");
  TH1F *hProton[2],   *hNeutron[2],   *hHeavy[2],   *hIon[2];
  TH1F *hProton_1[2], *hNeutron_1[2], *hHeavy_1[2], *hIon_1[2];
  for (int i=0; i<2; i++) {
    sprintf(title, "proton%i_%s%s%sGeV(%s)", i, target, list, ene, ctype);
    hProton[i] = (TH1F*)fout->FindObjectAny(title);
    sprintf(title, "proton%i_%s%s%sGeV", i, target, list, ene);
    hProton[i]->SetName(title);
    hProton[i]->SetTitle(title);

    hProton_1[i] = (TH1F*) hProton[i]->Clone();
    sprintf(title, "new_%s", title);
    hProton_1[i]->SetName(title);

    sprintf(title, "neutron%i_%s%s%sGeV(%s)", i, target, list, ene, ctype);
    hNeutron[i] = (TH1F*)fout->FindObjectAny(title);
    sprintf(title, "neutron%i_%s%s%sGeV", i, target, list, ene);
    hNeutron[i]->SetName(title);
    hNeutron[i]->SetTitle(title);

    hNeutron_1[i] = (TH1F*) hNeutron[i]->Clone();
    sprintf(title, "new_%s", title);
    hNeutron_1[i]->SetName(title);

    sprintf(title, "heavy%i_%s%s%sGeV(%s)", i, target, list, ene, ctype);
    hHeavy[i] = (TH1F*)fout->FindObjectAny(title);
    sprintf(title, "heavy%i_%s%s%sGeV", i, target, list, ene);
    hHeavy[i]->SetName(title);
    hHeavy[i]->SetTitle(title);

    hHeavy_1[i] = (TH1F*) hHeavy[i]->Clone();
    sprintf(title, "new_%s", title);
    hHeavy_1[i]->SetName(title);

    sprintf(title, "ion%i_%s%s%sGeV(%s)", i, target, list, ene, ctype);
    hIon[i] = (TH1F*)fout->FindObjectAny(title);
    sprintf(title, "ion%i_%s%s%sGeV", i, target, list, ene);
    hIon[i]->SetName(title);
    hIon[i]->SetTitle(title);

    hIon_1[i] = (TH1F*) hIon[i]->Clone();
    sprintf(title, "new_%s", title);
    hIon_1[i]->SetName(title);
  }

  int energy = atoi(ene);
  std::cout << "Energy " << energy << "\n";
  if (energy>=10) {
    hProton[0]->Rebin(5);   hProton[1]->Rebin(5);
    hProton_1[0]->GetXaxis()->SetRangeUser(0.0, 5.0);
    hProton_1[1]->GetXaxis()->SetRangeUser(0.0, 2.5);
    
    hNeutron[0]->Rebin(5);  hNeutron[1]->Rebin(5);
    hNeutron_1[0]->GetXaxis()->SetRangeUser(0.0, 5.0);
    hNeutron_1[1]->GetXaxis()->SetRangeUser(0.0, 2.5);

    hHeavy[0]->Rebin(5);    hHeavy[1]->Rebin(5);
    hHeavy_1[0]->GetXaxis()->SetRangeUser(0.0, 5.0);
    hHeavy_1[1]->GetXaxis()->SetRangeUser(0.0, 5.0);

    hIon[0]->GetXaxis()->SetRangeUser(0.0, 0.03);
    hIon[1]->GetXaxis()->SetRangeUser(0.0, 0.03);
  } else {
    
    hProton_1[0]->GetXaxis()->SetRangeUser(0.0, 1.0);
    hProton_1[1]->GetXaxis()->SetRangeUser(0.0, 0.5);

    hNeutron_1[0]->GetXaxis()->SetRangeUser(0.0, 1.0);
    hNeutron_1[1]->GetXaxis()->SetRangeUser(0.0, 0.5);
    
    hHeavy_1[0]->GetXaxis()->SetRangeUser(0.0, 5.0);
    hHeavy_1[1]->GetXaxis()->SetRangeUser(0.0, 5.0);

    hIon[0]->Rebin(5);      hIon[1]->Rebin(5);
    hIon[0]->GetXaxis()->SetRangeUser(0.0, 1.5);
    hIon[1]->GetXaxis()->SetRangeUser(0.0, 1.5);
  }
  
  for (int i=0; i<2; i++) {
    hProton[i]->SetFillColor(30);
    hNeutron[i]->SetFillColor(30);
    hHeavy[i]->SetFillColor(30);
    hIon[i]->SetFillColor(30);

    hProton_1[i]->SetFillColor(30);
    hNeutron_1[i]->SetFillColor(30);
    hHeavy_1[i]->SetFillColor(30);
    hIon_1[i]->SetFillColor(30);

    hProton[i]->GetXaxis()->SetRangeUser(0.0, energy);
    hNeutron[i]->GetXaxis()->SetRangeUser(0.0, energy);
    hHeavy[i]->GetXaxis()->SetRangeUser(0.0, energy);
  }

  sprintf(cname, "c_%s%s%sGeV_protons", target, list, ene);
  TCanvas *cc4 = new TCanvas(cname, cname, 800, 800);
  cc4->Divide(2,2);
  cc4->cd(1); hProton[0]->Draw();
  cc4->cd(2); hProton_1[0]->Draw();
  cc4->cd(3); hProton[1]->Draw();
  cc4->cd(4); hProton_1[1]->Draw();

  sprintf(cname, "c_%s%s%sGeV_neutrons", target, list, ene);
  TCanvas *cc5 = new TCanvas(cname, cname, 800, 800);
  cc5->Divide(2,2);
  cc5->cd(1); hNeutron[0]->Draw();
  cc5->cd(2); hNeutron_1[0]->Draw();
  cc5->cd(3); hNeutron[1]->Draw();
  cc5->cd(4); hNeutron_1[1]->Draw();

  sprintf(cname, "c_%s%s%sGeV_Heavy", target, list, ene);
  TCanvas *cc6 = new TCanvas(cname, cname, 800, 800);
  cc6->Divide(2,2);
  cc6->cd(1); hHeavy[0]->Draw();
  cc6->cd(2); hHeavy_1[0]->Draw();
  cc6->cd(3); hHeavy[1]->Draw();
  cc6->cd(4); hHeavy_1[1]->Draw();

  sprintf(cname, "c_%s%s%sGeV_Ion", target, list, ene);
  TCanvas *cc7 = new TCanvas(cname, cname, 800, 500);
  cc7->Divide(2,1);
  cc7->cd(1); hIon[0]->Draw();
  cc7->cd(2); hIon[1]->Draw();
}


void plotMultiplicity(char target[6], char list[20], char part[4], int ymax=25,
		      char dir[12]="histo", char g4ver[20]="G4.9.1.p01", 
		      bool flag=true) {

  setStyle();
  gStyle->SetOptTitle(0);
  
  char name[1024], sym[10];
  if      (part=="pim") sprintf(sym, "#pi^{-}");
  else if (part=="pip") sprintf(sym, "#pi^{+}");
  else                  sprintf(sym, "p");

  std::map<string, double> means_300=getMean(target,list,part,"300.0","Multi",dir);
  std::map<string, double> means_200=getMean(target,list,part,"200.0","Multi",dir);
  std::map<string, double> means_150=getMean(target,list,part,"150.0","Multi",dir);
  std::map<string, double> means_100=getMean(target,list,part,"100.0","Multi",dir);
  std::map<string, double> means_50 =getMean(target,list,part,"50.0", "Multi",dir);
  std::map<string, double> means_30 =getMean(target,list,part,"30.0", "Multi",dir);
  std::map<string, double> means_20 =getMean(target,list,part,"20.0", "Multi",dir);
  std::map<string, double> means_15 =getMean(target,list,part,"15.0", "Multi",dir);
  std::map<string, double> means_9  =getMean(target,list,part,"9.0",  "Multi",dir);
  std::map<string, double> means_7  =getMean(target,list,part,"7.0",  "Multi",dir);
  std::map<string, double> means_5  =getMean(target,list,part,"5.0",  "Multi",dir);
  std::map<string, double> means_3  =getMean(target,list,part,"3.0",  "Multi",dir);
  std::map<string, double> means_2  =getMean(target,list,part,"2.0",  "Multi",dir);
  std::map<string, double> means_1  =getMean(target,list,part,"1.0",  "Multi",dir);
  if (flag) {
    std::map<string, double> means_10 =getMean(target,list,part,"10.0", "Multi",dir);
    std::map<string, double> means_8  =getMean(target,list,part,"8.0",  "Multi",dir);
    std::map<string, double> means_6  =getMean(target,list,part,"6.0",  "Multi",dir);
    std::map<string, double> means_4  =getMean(target,list,part,"4.0",  "Multi",dir);
  }

  char ctype[20];
  std::vector<std::string> types   = types();
  std::vector<std::string> typeOld = typesOld();
  //  std::cout << "Number of types: " << types.size() << "\n";

  TGraph *gr[20];
  TLegend *leg = new TLegend(0.45, 0.53, 0.90, 0.90);
  char hdr[160];
  sprintf(hdr, "%s+%s (%s-%s)", sym, target, g4ver, list);
  leg->SetHeader(hdr);  leg->SetFillColor(10); leg->SetMargin(0.45);
  leg->SetTextSize(.027);
  sprintf(name, "c_%s_%sMultiplicity_%s", part,target,list);
  TCanvas *cc = new TCanvas(name, name, 700, 700);

  for (unsigned int ii=0; ii<=(types.size()); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", typeOld[ii-1].c_str());

    // std::cout<<"ii "<<ii<<"  ctype "<<ctype<<std::endl;

    string a(ctype);
    double vx[18], vy[18];
    int np=0;
    vx[np] = 300.0;  vy[np] = means_300[a]; np++;
    vx[np] = 200.0;  vy[np] = means_200[a]; np++;
    vx[np] = 150.0;  vy[np] = means_150[a]; np++;
    vx[np] = 100.0;  vy[np] = means_100[a]; np++;
    vx[np] = 50.0;   vy[np] = means_50[a];  np++;
    vx[np] = 30.0;   vy[np] = means_30[a];  np++;
    vx[np] = 20.0;   vy[np] = means_20[a];  np++;
    vx[np] = 15.0;   vy[np] = means_15[a];  np++;
    if (flag) { vx[np] = 10.0;   vy[np] = means_10[a];  np++;}
    vx[np] = 9.0;    vy[np] = means_9[a];   np++;
    if (flag) { vx[np] = 8.0;    vy[np] = means_8[a];   np++;}
    vx[np] = 7.0;    vy[np] = means_7[a];   np++;
    if (flag) { vx[np] = 6.0;    vy[np] = means_6[a];   np++;}
    vx[np] = 5.0;    vy[np] = means_5[a];   np++;
    if (flag && part != "pro") { vx[np] = 4.0;    vy[np] = means_4[a];   np++;}
    vx[np] = 3.0;    vy[np] = means_3[a];   np++;
    vx[np] = 2.0;    vy[np] = means_2[a];   np++;
    vx[np] = 1.0;    vy[np] = means_1[a];   np++;

    if (ii > 20 ) {
      std::cout << ctype;
      for (int ix=0; ix<np; ix++) std::cout << " " << vx[ix] << " " << vy[ix];
      std::cout << "\n";
    }

    gPad->SetLogx(1);
    gPad->SetGridx(1); gPad->SetGridy(1);
    gr[ii] = new TGraph(np, vx,vy);
    sprintf(name, "Multiplicity of secondaries %s-%s (%s %s)", sym, target, g4ver, list);
    gr[ii]->SetTitle(name);
    gr[ii]->GetXaxis()->SetTitle("Beam Momentum (GeV)");
    gr[ii]->GetYaxis()->SetTitle("Average Multiplicity");

    gr[ii]->SetMarkerStyle(mstyle[ii]);
    gr[ii]->SetMarkerSize(msiz[ii]);
    gr[ii]->SetMarkerColor(mcolor[ii]);
    gr[ii]->SetLineColor(lcolor[ii]);
    gr[ii]->SetLineStyle(lstyle[ii]);
    gr[ii]->SetLineWidth(lwidth[ii]); 

    gr[ii]->GetYaxis()->SetRangeUser(-0.2, ymax);
    if (ii>1) {
      sprintf (ctype, "%s", types[ii-1].c_str());
      leg->AddEntry(gr[ii], ctype, "lP");
    }
    if      (ii==2) gr[ii]->Draw("APl"); 
    else if (ii>2)  gr[ii]->Draw("Pl");
  }
  leg->Draw("same");
}

void plotMultiplicity(char target[6], char list[20], char ene[6], char part[4],
		      char dir[12]="histo", char g4ver[20]="G4.9.1.p01") {

  setStyle();
  gStyle->SetOptTitle(0);
  
  char name[1024], sym[10];
  if      (part=="pim") sprintf(sym, "#pi^{-}");
  else if (part=="pip") sprintf(sym, "#pi^{+}");
  else                  sprintf(sym, "p");

  std::vector<std::string> typeOld = typesOld();
  int energy = atoi(ene);
  
  char ofile[100];
  sprintf (ofile, "%s/histo_%s%s_%s_%sGeV.root", dir, target, list, part, ene);
  std::cout << "Input file " << ofile << "\n";
  TFile *fout = TFile::Open(ofile);
  fout->cd();

  char name[160], title[160], ctype[20], ytitle[20], cname[160];
  TH1I *hiMulti[20];
  for (unsigned int ii=0; ii<=(typeOld.size()); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", typeOld[ii-1].c_str());
    sprintf (name, "Multi%s%s%sGeV(%s)", target, list, ene, ctype);
    hiMulti[ii] = (TH1I*)fout->FindObjectAny(name);
    //    std::cout << ii << " (" << ctype << ") " << name << " " << hiMulti[ii] << "\n";
  }

  TCanvas *c[20];
  std::vector<std::string> types = types();
  for (unsigned int ii=0; ii<types.size(); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", types[ii-1].c_str());
    sprintf (cname, "Multiplicity (%s)", ctype);
    hiMulti[ii]->GetXaxis()->SetTitle(cname);
    hiMulti[ii]->SetMarkerStyle(mstyle[ii]);
    hiMulti[ii]->SetMarkerSize(msiz[ii]);
    hiMulti[ii]->SetMarkerColor(mcolor[ii]);
    hiMulti[ii]->SetLineColor(lcolor[ii]);
    hiMulti[ii]->SetLineStyle(lstyle[ii]);
    hiMulti[ii]->SetLineWidth(lwidth[ii]); 

    sprintf(cname, "c_%s%s_%s_%sGeV_Multiplicity(%s)", target, list, part, 
	    ene, ctype);
    c[ii] = new TCanvas(cname, cname, 800, 500);
    hiMulti[ii]->Draw();

    TLegend *leg = new TLegend(0.35, 0.80, 0.8, 0.87);
    char hdr[160];
    sprintf(hdr, "%s+%s at %s GeV (%s-%s)", sym, target, ene, g4ver, list);
    leg->SetHeader(hdr);  leg->SetFillColor(10); leg->SetMargin(0.45);
    leg->SetTextSize(.036); leg->Draw("same");
  }
}

void plotTotalKE(char target[6], char list[20], char part[4], 
		 char dir[12]="histo", char g4ver[20]="G4.9.1.p01",
		 bool flag=true) {

  setStyle();
  gStyle->SetOptTitle(0);

  char name[1024];
  char sym[10];
  if      (part=="pim") sprintf(sym, "#pi^{-}");
  else if (part=="pip") sprintf(sym, "#pi^{+}");
  else                  sprintf(sym, "p");

  std::map<string, double> means_300=getMean(target,list,part,"300.0","TotalKE",dir);
  std::map<string, double> means_200=getMean(target,list,part,"200.0","TotalKE",dir);
  std::map<string, double> means_150=getMean(target,list,part,"150.0","TotalKE",dir);
  std::map<string, double> means_100=getMean(target,list,part,"100.0","TotalKE",dir);
  std::map<string, double> means_50 =getMean(target,list,part,"50.0", "TotalKE",dir);
  std::map<string, double> means_30 =getMean(target,list,part,"30.0", "TotalKE",dir);
  std::map<string, double> means_20 =getMean(target,list,part,"20.0", "TotalKE",dir);
  std::map<string, double> means_15 =getMean(target,list,part,"15.0", "TotalKE",dir);
  std::map<string, double> means_9  =getMean(target,list,part,"9.0",  "TotalKE",dir);
  std::map<string, double> means_7  =getMean(target,list,part,"7.0",  "TotalKE",dir);
  std::map<string, double> means_5  =getMean(target,list,part,"5.0",  "TotalKE",dir);
  std::map<string, double> means_3  =getMean(target,list,part,"3.0",  "TotalKE",dir);
  std::map<string, double> means_2  =getMean(target,list,part,"2.0",  "TotalKE",dir);
  std::map<string, double> means_1  =getMean(target,list,part,"1.0",  "TotalKE",dir);
  if (flag) {
    std::map<string, double> means_10 =getMean(target,list,part,"10.0", "TotalKE",dir);
    std::map<string, double> means_8  =getMean(target,list,part,"8.0",  "TotalKE",dir);
    std::map<string, double> means_6  =getMean(target,list,part,"6.0",  "TotalKE",dir);
    std::map<string, double> means_4  =getMean(target,list,part,"4.0",  "TotalKE",dir);
  }

  char ctype[20];
  std::vector<std::string> types   = types();
  std::vector<std::string> typeOld = typesOld();
  //  std::cout << "Number of types " << types.size() << "\n";

  TGraph *gr[20];
  TLegend *leg = new TLegend(0.55, 0.45, 0.9, 0.80);
  char hdr[160];
  sprintf(hdr, "%s+%s (%s-%s)", sym, target, g4ver, list);
  leg->SetHeader(hdr);
  leg->SetFillColor(10);
  leg->SetMargin(0.45);
  leg->SetTextSize(.02);
  sprintf(name, "c_%s_%s_totalKE_%s", part,target,list);
  TCanvas *cc = new TCanvas(name, name, 700, 700);

  for (unsigned int ii=0; ii<=(types.size()); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", typeOld[ii-1].c_str());

    string a(ctype);
    //    std::cout<<a<<" "<< means_300[a]<<std::endl;
    double vx[18], vy[18];
    int np=0;
    vx[np] = 300.0;  vy[np] = means_300[a]; np++;
    vx[np] = 200.0;  vy[np] = means_200[a]; np++;
    vx[np] = 150.0;  vy[np] = means_150[a]; np++;
    vx[np] = 100.0;  vy[np] = means_100[a]; np++;
    vx[np] = 50.0;   vy[np] = means_50[a];  np++;
    vx[np] = 30.0;   vy[np] = means_30[a];  np++;
    vx[np] = 20.0;   vy[np] = means_20[a];  np++;
    vx[np] = 15.0;   vy[np] = means_15[a];  np++;
    if (flag) { vx[np] = 10.0;   vy[np] = means_10[a];  np++;}
    vx[np] = 9.0;    vy[np] = means_9[a];   np++;
    if (flag) { vx[np] = 8.0;    vy[np] = means_8[a];   np++;}
    vx[np] = 7.0;    vy[np] = means_7[a];   np++;
    if (flag) { vx[np] = 6.0;    vy[np] = means_6[a];   np++;}
    vx[np] = 5.0;    vy[np] = means_5[a];   np++;
    if (flag && part != "pro") { vx[np] = 4.0;    vy[np] = means_4[a];   np++;}
    vx[np] = 3.0;    vy[np] = means_3[a];   np++;
    vx[np] = 2.0;    vy[np] = means_2[a];   np++;
    vx[np] = 1.0;    vy[np] = means_1[a];   np++;

    for (int i=0; i<np; i++) vy[i] = vy[i]/vx[i];

    gPad->SetLogx(1);
    gPad->SetGridx(1);
    gPad->SetGridy(1);
    gr[ii] = new TGraph(np, vx,vy);
    sprintf(name, "KE carried by secondaries in %s-%s (%s)", sym, target, list);
    gr[ii]->SetTitle(name);
    gr[ii]->GetXaxis()->SetTitle("Beam Momentum (GeV)");
    gr[ii]->GetYaxis()->SetTitle("Mean Total KE/Beam Momentum");

    gr[ii]->SetMarkerStyle(mstyle[ii]);
    gr[ii]->SetMarkerSize(msiz[ii]);
    gr[ii]->SetMarkerColor(mcolor[ii]);
    gr[ii]->SetLineColor(lcolor[ii]);
    gr[ii]->SetLineStyle(lstyle[ii]);
    gr[ii]->SetLineWidth(lwidth[ii]); 

    gr[ii]->GetYaxis()->SetRangeUser(-0.02, 1.0);
    if (ii!= 0) sprintf (ctype, "%s", types[ii-1].c_str());
    if (ii!= 1) leg->AddEntry(gr[ii], ctype, "lP");
    if (ii==0)      gr[ii]->Draw("APl");
    else if (ii>1)  gr[ii]->Draw("Pl");
  }
  leg->Draw("same");
}

void plotKE(char target[6], char list[20], char ene[6], char part[4],
	    int typ=0, char dir[12]="histo", char g4ver[20]="G4.9.1.p01") {

  setStyle();
  gStyle->SetOptTitle(0);
  gStyle->SetOptLogy(1);

  char name[1024];
  char sym[10];
  if      (part=="pim") sprintf(sym, "#pi^{-}");
  else if (part=="pip") sprintf(sym, "#pi^{+}");
  else                  sprintf(sym, "p");

  std::vector<std::string> typeOld = typesOld();
  int energy = atoi(ene);
  int bins=energy/4;
  float ener = energy;
  std::cout << "Energy " << ener << "\n";
  
  char ofile[100];
  sprintf (ofile, "%s/histo_%s%s_%s_%sGeV.root", dir, target, list, part, ene);
  std::cout << "Input file " << ofile << "\n";
  TFile *fout = TFile::Open(ofile);
  fout->cd();

  char name[160], title[160], ctype[20], ytitle[20], cname[160], pre[10];
  TH1F *hiKE[20];
  if (typ == 0) sprintf (pre, "KE2");
  else          sprintf (pre, "TotalKE");
  for (unsigned int ii=0; ii<=(typeOld.size()); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", typeOld[ii-1].c_str());
    sprintf (name, "%s%s%s%sGeV(%s)", pre, target, list, ene, ctype);
    hiKE[ii] = (TH1F*)fout->FindObjectAny(name);
    //    std::cout << ii << " (" << ctype << ") " << name << " " << hiKE[ii] <<"\n";
  }

  TCanvas *c[25];
  std::vector<std::string> types = types();
  for (unsigned int ii=0; ii<types.size(); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", types[ii-1].c_str());
    if (typ == 0) sprintf (cname, "Kinetic Energy of %s (GeV)", ctype);
    else          sprintf (cname, "Total Kinetic Energy of %s (GeV)", ctype);
    hiKE[ii]->GetXaxis()->SetTitle(cname);
    hiKE[ii]->SetMarkerStyle(mstyle[ii]);
    hiKE[ii]->SetMarkerSize(msiz[ii]);
    hiKE[ii]->SetMarkerColor(mcolor[ii]);
    hiKE[ii]->SetLineColor(lcolor[ii]);
    hiKE[ii]->SetLineStyle(lstyle[ii]);
    hiKE[ii]->SetLineWidth(lwidth[ii]); 
    if (bins > 0) hiKE[ii]->Rebin(bins);
    hiKE[ii]->GetXaxis()->SetRangeUser(0.0, ener);

    sprintf(cname, "c_%s%s_%s_%sGeV_%s(%s)", target,list,part,ene,pre,ctype);
    c[ii] = new TCanvas(cname, cname, 800, 500);
    hiKE[ii]->Draw();

    TLegend *leg = new TLegend(0.35, 0.80, 0.8, 0.87);
    char hdr[160];
    sprintf(hdr, "%s+%s at %s GeV (%s-%s)", sym, target, ene, g4ver, list);
    leg->SetHeader(hdr);  leg->SetFillColor(10); leg->SetMargin(0.45);
    leg->SetTextSize(.036); leg->Draw("same");
  }

  TLegend *leg1 = new TLegend(0.50, 0.75, 0.90, 0.90);
  if (typ == 0) sprintf (cname, "Kinetic Energy (GeV)");
  else          sprintf (cname, "Total Kinetic Energy (GeV)");
  hiKE[6]->GetXaxis()->SetTitle(cname);
  char hdr[160];
  sprintf(hdr, "%s+%s at %s GeV (%s-%s)", sym, target, ene, g4ver, list);
  leg1->SetHeader(hdr);  leg1->SetFillColor(10); leg1->SetMargin(0.45);
  sprintf(cname, "c_%s%s_%s_%sGeV_%s(Pion)", target,list,part,ene,pre);
  leg1->SetTextSize(.030); 
  c[19] = new TCanvas(cname, cname, 800, 500);
  hiKE[6]->Draw(); sprintf (ctype, "%s", types[5].c_str()); leg1->AddEntry(hiKE[6], ctype, "l");
  hiKE[5]->Draw("same"); sprintf (ctype, "%s", types[4].c_str()); leg1->AddEntry(hiKE[5], ctype, "l");
  hiKE[4]->Draw("same"); sprintf (ctype, "%s", types[3].c_str()); leg1->AddEntry(hiKE[4], ctype, "l"); leg->Draw("same");

  TLegend *leg2 = new TLegend(0.50, 0.75, 0.90, 0.90);
  if (typ == 0) sprintf (cname, "Kinetic Energy (GeV)");
  else          sprintf (cname, "Total Kinetic Energy (GeV)");
  hiKE[7]->GetXaxis()->SetTitle(cname);
  sprintf(hdr, "%s+%s at %s GeV (%s-%s)", sym, target, ene, g4ver, list);
  leg2->SetHeader(hdr);  leg2->SetFillColor(10); leg2->SetMargin(0.45);
  sprintf(cname, "c_%s%s_%s_%sGeV_%s(Kaon)", target,list,part,ene,pre);
  leg2->SetTextSize(.030); 
  c[20] = new TCanvas(cname, cname, 800, 500);
  hiKE[7]->Draw(); sprintf (ctype, "%s", types[6].c_str()); leg2->AddEntry(hiKE[7], ctype, "l");
  hiKE[8]->Draw("same"); sprintf (ctype, "%s", types[7].c_str()); leg2->AddEntry(hiKE[8], ctype, "l");
  hiKE[9]->Draw("same"); sprintf (ctype, "%s", types[8].c_str()); leg2->AddEntry(hiKE[9], ctype, "l"); leg2->Draw("same");

  TLegend *leg3 = new TLegend(0.50, 0.75, 0.90, 0.90);
  if (typ == 0) sprintf (cname, "Kinetic Energy (GeV)");
  else          sprintf (cname, "Total Kinetic Energy (GeV)");
  hiKE[12]->GetXaxis()->SetTitle(cname);
  sprintf(hdr, "%s+%s at %s GeV (%s-%s)", sym, target, ene, g4ver, list);
  leg3->SetHeader(hdr);  leg3->SetFillColor(10); leg3->SetMargin(0.45);
  sprintf(cname, "c_%s%s_%s_%sGeV_%s(Nucleon)", target,list,part,ene,pre);
  leg3->SetTextSize(.030); 
  c[21] = new TCanvas(cname, cname, 800, 500);
  hiKE[12]->Draw(); sprintf (ctype, "%s", types[11].c_str()); leg3->AddEntry(hiKE[12], ctype, "l");
  hiKE[11]->Draw("same"); sprintf (ctype, "%s", types[10].c_str()); leg3->AddEntry(hiKE[11], ctype, "l"); leg3->Draw("same");
}

void printMeans(std::map<string, double> means) {

  std::map<string, double>::iterator iter;
  for( iter = means.begin(); iter != means.end(); iter++ ) {
    std::cout << (*iter).first << " " << (*iter).second << "\n";
  }
}

std::map<string, double> getMean(char target[6], char list[20], char part[5], 
				 char ene[6], char ctyp0[10]="Multi",
				 char dir[12]="histo") {

  std::vector<std::string> types = typesOld();
  std::map<string, double> means;
  
  char ofile[100];
  sprintf (ofile, "%s/histo_%s%s_%s_%sGeV.root", dir, target, list, part, ene);
  std::cout << "Input File: " << ofile << "\n";
  TFile *fout = TFile::Open(ofile);
  fout->cd();

  TH1I *hi[20];
  char name[160], title[160], ctype[20];

  for (unsigned int ii=0; ii<=(types.size()); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", types[ii-1].c_str());

    sprintf (name, "%s%s%s%sGeV(%s)", ctyp0, target, list, ene, ctype);
    hi[ii] = (TH1I*)fout->FindObjectAny(name);
    //    std::cout << "Histo " << ii << " Name " << name << " " << hi[ii] << " " << hi[ii]->GetMean() << "\n";
    
    string a(ctype);
    means[a] = hi[ii]->GetMean();
  }

  //  printMeans(means);

  return means;
}

std::vector<std::string> types() {

  std::vector<string> tmp;
  tmp.push_back("Photon/Neutrino");     // 1
  tmp.push_back("e^{-}");               // 2
  tmp.push_back("e^{+}");               // 3 
  tmp.push_back("#pi^{0}");             // 4 
  tmp.push_back("#pi^{-}");             // 5
  tmp.push_back("#pi^{+}");             // 6
  tmp.push_back("K^{-}");               // 7
  tmp.push_back("K^{+}");               // 8
  tmp.push_back("K^{0}");               // 9
  tmp.push_back("AntiProton");          // 10
  tmp.push_back("p");                   // 11
  tmp.push_back("n");                   // 12
  tmp.push_back("Heavy Hadrons");       // 13
  tmp.push_back("Ions");                // 14

  return tmp;
}

std::vector<std::string> typesOld() {

  std::vector<string> tmp;
  tmp.push_back("Photon/Neutrino");     // 1
  tmp.push_back("Electron");            // 2
  tmp.push_back("Positron");            // 3 
  tmp.push_back("Pizero");              // 4 
  tmp.push_back("Piminus");             // 5
  tmp.push_back("Piplus");              // 6
  tmp.push_back("Kminus");              // 7
  tmp.push_back("Kiplus");              // 8
  tmp.push_back("Kzero");               // 9
  tmp.push_back("AntiProton");          // 10
  tmp.push_back("Proton");              // 11
  tmp.push_back("Neutron/AntiNeutron"); // 12
  tmp.push_back("Heavy Hadrons");       // 13
  tmp.push_back("Ions");                // 14

  return tmp;
}

std::vector<double> massScan() {

  std::vector<double> tmp;
  tmp.push_back(0.01);
  tmp.push_back(1.00);
  tmp.push_back(135.0);
  tmp.push_back(140.0);
  tmp.push_back(495.0);
  tmp.push_back(500.0);
  tmp.push_back(938.5);
  tmp.push_back(940.0);
  tmp.push_back(1850.0);
  std::cout << tmp.size() << " Mass regions for prtaicles: ";
  for (unsigned int i=0; i<tmp.size(); i++) {
    std::cout << tmp[i];
    if (i == tmp.size()-1) std::cout << " MeV\n";
    else                   std::cout << ", ";
  }
  return tmp;
}

void setStyle() {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetTitleOffset(1.2,"Y");  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(1);

}
