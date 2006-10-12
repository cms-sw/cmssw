void DoCompare( char* Sample ){

 static const int NHisto = 1;
 static const int NHisto2 = 4;
 // static const int NHisto3 = 2;

 TText* te = new TText();
 te->SetTextSize(0.1);

 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

 char*  reffilename  = "${REFFILE}";//"../data/EcalSimHitHisto_30GeV.root";
 char*  curfilename  = "${CURFILE}";//"../data/EcalSimHitHisto_30GeV.root";

 TFile * reffile = new TFile(reffilename);
 TFile * curfile = new TFile(curfilename);

 //1-Dimension Histogram
 char* label[NHisto];
 label[0] = "nvtx";

 TH1F* htemp1[NHisto];
 TH1F* htemp2[NHisto];
 for ( int i = 0; i< NHisto ; i++ ) {
   char title[50];
   TCanvas c1;
   htemp1[i]  = dynamic_cast<TH1F*>(reffile->Get(label[i]));
   htemp2[i]  = dynamic_cast<TH1F*>(curfile->Get(label[i]));
   if( htemp1[i] == 0 || htemp2[i] == 0) continue;
   htemp1[i]->SetLineColor(2);
   htemp2[i]->SetLineColor(4);
   htemp1[i]->SetLineStyle(3);
   htemp2[i]->SetLineStyle(5);
   if (i>14 && i<19 || i>29 && i< 34 || i == 13) c1.SetLogy();

   htemp1[i]->Draw();
   htemp2[i]->Draw("Same"); 
   myPV->PVCompute(htemp1[i],htemp2[i], te);
   sprintf(title,"%s%s%s", Sample, label[i],".eps");
   c1.Print(title);
 }
 
}


