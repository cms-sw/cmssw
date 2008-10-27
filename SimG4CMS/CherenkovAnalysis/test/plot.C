//________________________________________________________________________________________
int plot( TString fileName = "px-00.txt", bool redraw = true )
{
  if ( redraw ) {
   TCanvas* myCanvas = new TCanvas("myCanvas","Canvas",10,10,800,600);
   myCanvas->cd(1);
  }

  TTree* tree = new TTree("tree","Cherenkov photons");
  std::cout << tree->ReadFile(fileName,"px/D:py:pz:x:y:z") 
            << " lines read from " << fileName << std::endl;
  tree->Draw("x");
  
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
                                                                        
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  TCanvas* myCanvas = new TCanvas("myCanvas","Canvas",10,10,800,800);
  myCanvas->Divide(2,5);
  myCanvas->SetLogy(1);
  TH2F* range = new TH2F("range","Cherenkov photons",2,-40,40,2,-12,12);

  const int nfiles = 9;
  TString fileRoot("output");
  char fileName[50];

  // Draw angle = 0 twice
  TTree* tree0 = new TTree("tree0","Cherenkov photons");
  std::cout << tree0->ReadFile("output-1.txt","px/D:py:pz:x:y:z") 
            << " lines read from output-1.txt" << std::endl;
  for ( int i=1; i<=2; ++i ) {
    myCanvas->cd(i);
    range->Draw();
    tree0->Draw("y:x","","same");
  }
  
  // Draw other angles
  for ( int ifile = 1; ifile<nfiles; ++ifile ) {

    int ican = 2*ifile+1-(ifile/5)*7; // Sorry for that...
    std::cout << ican << std::endl;
    myCanvas->cd( ican );
    sprintf(fileName,"%s-%1d.txt",fileRoot.Data(),ifile+1);
    TTree* tree = new TTree("tree","Cherenkov photons");
    std::cout << tree->ReadFile(fileName,"px/D:py:pz:x:y:z") 
              << " lines read from " << fileName << std::endl;

    range->Draw();
    tree->Draw("y:x","","same");

  }

  //   int colors[] = { kRed-1, kBlue-1, kGreen-1, kMagenta-1, kYellow-3 };


  return 0;

}
