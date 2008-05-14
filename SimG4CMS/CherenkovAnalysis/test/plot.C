//________________________________________________________________________________________
int plot( TString fileName = "Cherenkov.root" )
{
   TCanvas* myCanvas = new TCanvas("myCanvas","Canvas",10,10,600,900);
   myCanvas->Divide(1,2);

   TFile* f = new TFile(fileName);
   f->cd("analyzer");

   myCanvas->cd(1);
   hEnergy->Draw();
   myCanvas->cd(2);
   hTimeStructure->Draw();

   return 0;
}


//________________________________________________________________________________________
int trackAngle( TString fileName = "simevent.root" )
{

  TFile* f = new TFile(fileName);
  TTree* events = f->Get("Events");

  events->SetAlias("position","SimTracks_g4SimHits__CaloTest.obj.tkposition.fCoordinates");
  events->SetAlias("momentum","SimTracks_g4SimHits__CaloTest.obj.theMomentum.fCoordinates");

  TCanvas* myCanvas = new TCanvas("myCanvas","Canvas",10,10,600,900);
  myCanvas->Divide(1,2);
  
  myCanvas->cd(1);
  events->Draw("position.fY");
  myCanvas->cd(2);
  events->Draw("TMath::ATan(momentum.fX/momentum.fY)*180/TMath::Pi()");

  return 0;
}


//________________________________________________________________________________________
int getAngleParams( double angle )
{

  double meanY = 0.9; // Y position of gun - (width crystal)/2

  double radAngle = angle*TMath::Pi()/180.;

  double meanX = -meanY*TMath::Tan( radAngle );
  
  double phi = TMath::PiOver2()-radAngle;

  std::cout << "    replace VtxSmeared.MeanX = " << meanX << std::endl;
  cout.precision(16);
  std::cout << "    replace FlatRandomEGunSource.PGunParameters.MinPhi = " << phi << std::endl;
  std::cout << "    replace FlatRandomEGunSource.PGunParameters.MaxPhi = " << phi << std::endl;

  return 0;

}

//________________________________________________________________________________________
int plotAll() {

   TCanvas* myCanvas = new TCanvas("myCanvas","Canvas",10,10,600,900);
   myCanvas->Divide(1,2);

   const int nfiles = 5;
   TString fileNames[] = {
     "files/Cherenkov-e10-a0.root",
     "files/Cherenkov-e30-a0.root",
     "files/Cherenkov-e50-a0.root",
     "files/Cherenkov-e70-a0.root",
     "files/Cherenkov-e100-a0.root"
   }

   int colors[] = { kRed-1, kBlue-1, kGreen-1, kMagenta-1, kYellow-3 };
   TFile* files[nfiles];
   
   // Draw energy deposit
   TPad* pad1 = myCanvas->cd(1);
//    pad1->Divide(2,1);
//    pad1->cd(1);
   TLegend* legend1 = new TLegend(0.6,0.5,0.89,0.85,"","brNDC");
   TGraph* maxGraph = new TGraph(nfiles+1);
   maxGraph->SetPoint(0,0,0);
   legend1->SetBorderSize(1);
   for ( int i=0; i<nfiles; ++i ) {
     files[i] = new TFile(fileNames[i]);
     TH1F* hist = files[i]->Get("analyzer/hEnergy");
     hist->SetLineColor(colors[i]);
     if ( !i ) {
       hist->GetXaxis()->SetTitle("Total energy [GeV]");
       hist->Draw();
     } else 
       hist->Draw("same");
     TObjArray *subStrL = TPRegexp("e(\\d+)").MatchS(fileNames[i]);
     const TString subStr = ((TObjString *)subStrL->At(1))->GetString();
     legend1->AddEntry(hist,subStr+" GeV e^{#pm}","l");
     maxGraph->SetPoint(i+1,
                        atof(subStr.Data()),
                        hist->GetBinCenter(hist->GetMaximumBin()));
   }
   legend1->Draw();
   
//    maxGraph->SetMarkerStyle(20);
//    maxGraph->SetMarkerColor(kRed);
//    maxGraph->SetLineColor(kRed);
//    maxGraph->SetTitle("Most probable value vs. energy ; Energy [GeV] ; MPV [GeV] ");
//    pad1->cd(2); maxGraph->Draw("AP");

   // Draw "time structure"
   myCanvas->cd(2);
   TLegend* legend2 = new TLegend(0.6,0.5,0.89,0.85,"","brNDC");
   legend2->SetBorderSize(1);
   for ( int i=nfiles-1; i>=0; --i ) {
     TH1F* hist = (TH1F*)files[i]->Get("analyzer/hTimeStructure");
     hist->SetLineColor(colors[i]);
     if ( i == nfiles-1 ) {
       hist->GetXaxis()->SetTitle("Time [ns]");
       hist->GetYaxis()->SetTitle("Summed energy [GeV]");
       hist->Draw();
     } else 
       hist->Draw("same");
     TObjArray *subStrL = TPRegexp("e(\\d+)").MatchS(fileNames[i]);
     const TString subStr = ((TObjString *)subStrL->At(1))->GetString();
     legend2->AddEntry(hist,subStr+" GeV e^{#pm}","l");
   }
   legend2->Draw();

   return 0;

}
