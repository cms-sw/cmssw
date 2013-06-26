#include "TStyle.h"

void plot(char type[10]="TTBar", char hist0[2]="C", int bin=5, int mode=0,
	  double xmax=-1., double ymax=-1., char fileps[40]="") {

  char file[60], hist[30], codes[6], code;
  //  sprintf(codes, "453216");
  sprintf (codes, "786519");
  sprintf (file, "fieldStudy%s.root", type);
  TFile* File = new TFile(file);

  int nhist=6;
  if      (mode==1) nhist=3;
  else if (mode==2) nhist=6;

  TCanvas *myc = new TCanvas("myc","",800,600);

  TH1F* hi[6];
  gPad->SetLogy(1);

  for (int ih=0; ih<nhist; ih++) {
    if (mode == 1) {
      if      (ih==1) sprintf (hist, "DQMData/StepC%s",  hist0);
      else if (ih==2) sprintf (hist, "DQMData/StepN%s",  hist0);
      else            sprintf (hist, "DQMData/Step%s",   hist0);
    } else if (mode == 2) {
      if      (ih==1) sprintf (hist, "DQMData/StepE%s",  hist0);
      else if (ih==2) sprintf (hist, "DQMData/StepG%s",  hist0);
      else if (ih==3) sprintf (hist, "DQMData/StepCH%s", hist0);
      else if (ih==4) sprintf (hist, "DQMData/StepNH%s", hist0);
      else if (ih==5) sprintf (hist, "DQMData/StepMu%s", hist0);
      else            sprintf (hist, "DQMData/Step%s",   hist0);
     } else if (mode == 3) {
      code = codes[ih];
      sprintf (hist, "DQMData/Step%s%s",  hist0, code);
    } else {
      code = codes[ih];
      sprintf (hist, "DQMData/Step%s", code);
    }
    hi[ih] = (TH1F*) File->Get(hist);
    hi[ih]->Rebin(bin);
    hi[ih]->SetLineStyle(ih+1);
    hi[ih]->SetLineWidth(2);
    hi[ih]->SetLineColor(ih+1);
    if (xmax > 0) hi[ih]->GetXaxis()->SetRangeUser(0.,xmax);
    if (ymax > 0) hi[ih]->SetMaximum(ymax);
    if (ih == 0) {
      hi[ih]->GetXaxis()->SetTitle("Step Size (mm)");
      hi[ih]->Draw("HIST");
    } else 
      hi[ih]->Draw("HIST same");
  }

  TLegend* leg = new TLegend(0.55,0.70,0.90,0.90);
  char det[20], header[30];
  if      (hist0 == codes[1]) sprintf (det, "Tracker");
  else if (hist0 == codes[2]) sprintf (det, "Muon");
  else if (hist0 == codes[3]) sprintf (det, "BeamPipe");
  else if (hist0 == codes[4]) sprintf (det, "Forward");
  else if (hist0 == codes[5]) sprintf (det, "Others");
  else if (hist0 == codes[0]) sprintf (det, "Calo");
  else if (hist0 == "C")      sprintf (det, "Charged");
  else if (hist0 == "N")      sprintf (det, "Neutral");
  else if (hist0 == "E")      sprintf (det, "Electron");
  else if (hist0 == "G")      sprintf (det, "Photon");
  else if (hist0 == "CH")     sprintf (det, "Charged Hadron");
  else if (hist0 == "NH")     sprintf (det, "Neutral Hadron");
  else if (hist0 == "Mu")     sprintf (det, "Muon");
  else if (hist0 == "0")      sprintf (det, "All Volumes");
  else                        sprintf (det, "Unknown");
  if (mode > 0 && mode <= 3)  sprintf (header, "%s Sample (%s)", type, det);
  else                        sprintf (header, "%s Sample", type);
  leg->SetHeader(header);
  for (int ih=0; ih<nhist; ih++) {
    if (mode == 1) {
      if      (ih==1) sprintf (hist, "Charged");
      else if (ih==2) sprintf (hist, "Neutral");
      else            sprintf (hist, "All");
    } else if (mode == 2) {
      if      (ih==1) sprintf (hist, "Electron");
      else if (ih==2) sprintf (hist, "Photon");
      else if (ih==3) sprintf (hist, "Charged Hadron");
      else if (ih==4) sprintf (hist, "Neutral Hadron");
      else if (ih==5) sprintf (hist, "Muon");
      else            sprintf (hist, "All");
    } else {
      if      (ih==1) sprintf (hist, "Tracker");
      else if (ih==2) sprintf (hist, "Muon");
      else if (ih==3) sprintf (hist, "BeamPipe");
      else if (ih==4) sprintf (hist, "Forward");
      else if (ih==5) sprintf (hist, "Other");
      else            sprintf (hist, "Calo");
    }
    leg->AddEntry(hi[ih],hist,"F");
  }
  leg->Draw();

  if (fileps != "") myc->SaveAs(fileps);
  return;
}
