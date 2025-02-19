void DoCompare( char* Sample ){

  static const int NHisto = 12;
  static const int NPages = 12;

  TText* te = new TText();
  te->SetTextSize(0.1);
  
  gROOT->ProcessLine(".x HistoCompare.C");
  HistoCompare * myPV = new HistoCompare();
  
  char*  reffilename  = "${REFFILE}";
  char*  curfilename  = "${CURFILE}";
  
  TFile * reffile = new TFile(reffilename);
  TFile * curfile = new TFile(curfilename);
  
  //1-Dimension Histogram
  char* label[NHisto];
  label[0] = "nbvtx";
  label[1] = "nbtksinvtx";
  label[2] = "resx";
  label[3] = "resy";
  label[4] = "resz";
  label[5] = "pullx";
  label[6] = "pully";
  label[7] = "pullz";
  label[8] = "vtxchi2";
  label[9] = "vtxndf";
  label[10] = "tklinks";
  label[11] = "nans";
  
  TH1F* htemp1[NHisto];
  TH1F* htemp2[NHisto];
  TCanvas* c1 = new TCanvas();
  c1->Divide(4, 3);
  for ( int i = 0; i < NHisto ; i++ ) {
    char title[50];
    htemp1[i]  = dynamic_cast<TH1F*>(reffile->Get(label[i]));
    htemp2[i]  = dynamic_cast<TH1F*>(curfile->Get(label[i]));
    if( htemp1[i] == 0 || htemp2[i] == 0) continue;
    htemp1[i]->SetLineColor(2);
    htemp2[i]->SetLineColor(4);
    htemp1[i]->SetLineStyle(3);
    htemp2[i]->SetLineStyle(5);
    if (i>14 && i<19 || i>29 && i< 34 || i == 13) c1->SetLogy();
    
    c1->cd(i % NPages + 1);
    htemp1[i]->Draw();
    htemp2[i]->Draw("Same"); 
    myPV->PVCompute(htemp1[i],htemp2[i], te);
    
    if (((i+1) % NPages) == 0) {
      sprintf(title,"%s%s%s", Sample, label[i], ".eps");
      c1->Print(title);
    }
  }
}
