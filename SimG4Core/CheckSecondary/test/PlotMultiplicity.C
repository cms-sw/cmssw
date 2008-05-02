#include <iostream>
#include <vector>
#include <map>


void PlotMultiplicity() {

  char element[6];
  sprintf(element, "PbWO4");
  PlotMultiplicity(element);

  sprintf(element, "Brass");
  PlotMultiplicity(element);

}

void PlotMultiplicity(char element[6]) {

  char list[10], name[50];
  sprintf(list, "QGSP");
  
  std::map<string, double> means_300 = getMean(element, "QGSP", "300.0");
  printMeans(means_300);
  
  std::map<string, double> means_200 = getMean(element, "QGSP", "200.0");
  printMeans(means_200);

  std::map<string, double> means_150 = getMean(element, "QGSP", "150.0");
  printMeans(means_150);

  std::map<string, double> means_100 = getMean(element, "QGSP", "100.0");
  printMeans(means_100);

  std::map<string, double>  means_50 = getMean(element, "QGSP", "50.0");
  printMeans(means_50);

  std::map<string, double>  means_30 = getMean(element, "QGSP", "30.0");
  printMeans(means_30);

  std::map<string, double>  means_20 = getMean(element, "QGSP", "20.0");
  printMeans(means_30);

  std::map<string, double>  means_15 = getMean(element, "QGSP", "15.0");
  printMeans(means_15);

  std::map<string, double>   means_9 = getMean(element, "QGSP", "9.0");
  printMeans(means_9);

  std::map<string, double>   means_7 = getMean(element, "QGSP", "7.0");
  printMeans(means_7);

  std::map<string, double>   means_5 = getMean(element, "QGSP", "5.0");
  printMeans(means_5);

  std::map<string, double>   means_3 = getMean(element, "QGSP", "3.0");
  printMeans(means_3);

  std::map<string, double>   means_2 = getMean(element, "QGSP", "2.0");
  printMeans(means_2);

  std::map<string, double>   means_1 = getMean(element, "QGSP", "1.0");
  printMeans(means_1);

  char ctype[20];
  std::vector<double> masses = massScan();

  setStyle();
  
  TGraph *gr[15];
  TLegend *leg = new TLegend(0.2, 0.5, 0.6, 0.85);
  leg->SetFillColor(10);
  sprintf(name, "c_%sMultiplicity", element);
  TCanvas *cc = new TCanvas(name, name, 700, 700);

  for (unsigned int ii=0; ii<=(masses.size())+1; ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else if (ii == 1) sprintf (ctype, "Photons");
    else if (ii == 2) sprintf (ctype, "Electrons/Positrons");
    else if (ii == 3) sprintf (ctype, "Neutral Pions");
    else if (ii == 4) sprintf (ctype, "Charged Pions");
    else if (ii == 5) sprintf (ctype, "Charged Kaons");
    else if (ii == 6) sprintf (ctype, "Neutral Kaons");
    else if (ii == 7) sprintf (ctype, "Protons/Antiportons");
    else if (ii == 8) sprintf (ctype, "Neutrons");
    else if (ii == 9) sprintf (ctype, "Heavy hadrons");
    else              sprintf (ctype, "Ions");

    string a(ctype);
    double vx[14], vy[14];
    vx[0]  = 300.0;  vy[0]  = means_300[a];
    vx[1]  = 200.0;  vy[1]  = means_200[a];
    vx[2]  = 150.0;  vy[2]  = means_150[a];
    vx[3]  = 100.0;  vy[3]  = means_100[a];
    vx[4]  = 50.0;   vy[4]  = means_50[a];
    vx[5]  = 30.0;   vy[5]  = means_30[a];
    vx[6]  = 20.0;   vy[6]  = means_20[a];
    vx[7]  = 15.0;   vy[7]  = means_15[a];
    vx[8]  = 9.0;    vy[8]  = means_9[a];
    vx[9]  = 7.0;    vy[9]  = means_7[a];
    vx[10] = 5.0;    vy[10] = means_5[a];
    vx[11] = 3.0;    vy[11] = means_3[a];
    vx[12] = 2.0;    vy[12] = means_2[a];
    vx[13] = 1.0;    vy[13] = means_1[a];

    gPad->SetLogx(1);
    gPad->SetGridx(1);
    gPad->SetGridy(1);
    gr[ii] = new TGraph(14, vx,vy);
    sprintf(name, "Multiplicity of particles in %s (%s)", element, list);
    gr[ii]->SetTitle(name);
    gr[ii]->GetXaxis()->SetTitle("Beam Energy (GeV)");
    gr[ii]->SetMarkerStyle(20);
    gr[ii]->SetMarkerColor(ii+1);
    gr[ii]->SetLineColor(ii+1);
    if(ii+1 == 3){ 
      gr[ii]->SetMarkerStyle(21);
      gr[ii]->SetMarkerColor(ii); 
      gr[ii]->SetLineColor(ii); 
      gr[ii]->SetLineStyle(2); 
      gr[ii]->SetLineWidth(2);
    }
    if(ii+1 == 5){ 
      gr[ii]->SetMarkerStyle(21);
      gr[ii]->SetMarkerColor(ii); 
      gr[ii]->SetLineColor(ii); 
      gr[ii]->SetLineStyle(2); 
      gr[ii]->SetLineWidth(2);
    }
    if(ii+1 == 7){ 
      gr[ii]->SetMarkerStyle(21);
      gr[ii]->SetMarkerColor(ii); 
      gr[ii]->SetLineColor(ii); 
      gr[ii]->SetLineStyle(2); 
      gr[ii]->SetLineWidth(2);
    }
    if(ii+1 == 9){ 
      gr[ii]->SetMarkerStyle(21);
      gr[ii]->SetMarkerColor(ii); 
      gr[ii]->SetLineColor(ii); 
      gr[ii]->SetLineStyle(2); 
      gr[ii]->SetLineWidth(2);
    }
    if(ii+1 == 10) {
      gr[ii]->SetMarkerColor(28);
      gr[ii]->SetLineColor(28);
    }
    if(ii+1 == 11){ 
      gr[ii]->SetMarkerStyle(21);
      gr[ii]->SetMarkerColor(28); 
      gr[ii]->SetLineColor(28); 
      gr[ii]->SetLineStyle(2); 
      gr[ii]->SetLineWidth(2);
    }
    
    gr[ii]->GetYaxis()->SetRangeUser(-0.1, 14);
    if(ii>0)leg->AddEntry(gr[ii], ctype, "lP");
    if(ii==1)      gr[ii]->Draw("APC");
    else if(ii>1)  gr[ii]->Draw("PC");
  }

  leg->Draw("same");
  
}

void printMeans(std::map<string, double> means){

  std::map<string, double>::iterator iter;
  for( iter = means.begin(); iter != means.end(); iter++ ) {
    std::cout << (*iter).first << " " << (*iter).second << "\n";
  }
}

std::map<string, double> getMean(char element[6], char list[10], char ene[6]){

  std::vector<double> masses = massScan();

  std::map<string, double> means;
  
  char ofile[50];
  sprintf (ofile, "histo_%s%s%sGeV.root", element, list, ene);
  std::cout<<ofile<<std::endl;
  TFile *fout = TFile::Open(ofile);
  fout->cd();
  //  fout->ls();

  TH1I *hiMulti[15];
  char name[60], title[160], ctype[20], cname[160];

  sprintf(cname, "c_%s%s%sGeV", element, list, ene);
  //  TCanvas *cc3 = new TCanvas(cname, cname, 800, 800);

  for (unsigned int ii=0; ii<=(masses.size())+1; ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else if (ii == 1) sprintf (ctype, "Photons");
    else if (ii == 2) sprintf (ctype, "Electrons/Positrons");
    else if (ii == 3) sprintf (ctype, "Neutral Pions");
    else if (ii == 4) sprintf (ctype, "Charged Pions");
    else if (ii == 5) sprintf (ctype, "Charged Kaons");
    else if (ii == 6) sprintf (ctype, "Neutral Kaons");
    else if (ii == 7) sprintf (ctype, "Protons/Antiportons");
    else if (ii == 8) sprintf (ctype, "Neutrons");
    else if (ii == 9) sprintf (ctype, "Heavy hadrons");
    else              sprintf (ctype, "Ions");

    sprintf (name, "Multi%s%s%sGeV(%s)", element, list, ene, ctype);
    //    std::cout<<name<<std::endl;
    hiMulti[ii] = (TH1I*)fout->FindObjectAny(name);

    hiMulti[ii]->SetLineColor(ii+1);
    if(ii>=9) hiMulti[ii]->SetLineColor(ii+2);
    //    if(ii==0) hiMulti[ii]->Draw();
    //    else      hiMulti[ii]->Draw("sames");
    
    string a(ctype);
    means[a] = hiMulti[ii]->GetMean();
  } //int ii for types

  std::cout<<"means size "<<means.size()<<std::endl;

  /*
  std::map<string, double>::iterator iter;
  for( iter = means.begin(); iter != means.end(); iter++ ) {
    std::cout<<(*iter).first << " "<<(*iter).second << std::endl;
  }
  */

  return means;
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
  gStyle->SetTitleOffset(1.6,"Y");  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(1);

}
