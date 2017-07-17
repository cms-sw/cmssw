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
  std::cout << "    replace FlatRandomEGunProducer.PGunParameters.MinPhi = " << phi << std::endl;
  std::cout << "    replace FlatRandomEGunProducer.PGunParameters.MaxPhi = " << phi << std::endl;

  return 0;

}

//________________________________________________________________________________________
int plotAll() {
                                                                        
  gStyle->SetOptStat(0);
  gStyle->SetOptDate(0);
  gStyle->SetOptTitle(0);
  TCanvas* myCanvas = new TCanvas("myCanvas","Canvas",10,10,800,800);
  myCanvas->Divide(2,5);
  TH2F* range = new TH2F("range","Cherenkov photons ; x [mm]; y [mm]",2,-50,50,2,-12,12);
  //TH2F* range = new TH2F("range","Cherenkov photons ; #(photons)/step",2,0,300,2,0,300);
  range->GetXaxis()->SetTitleSize(0.09);
  range->GetXaxis()->SetTitleOffset(-0.5);
  range->GetYaxis()->SetTitleSize(0.09);
  range->GetYaxis()->SetTitleOffset(0.5);

  const int nfiles = 9;
  TString fileRoot("output");
  char fileName[50];

  // Draw angle = 0 twice
  TLatex* t = new TLatex();
  char angle[20];
  TFile* file = new TFile("output-1.root");
  TTree* tree0 = file->Get("g4SimHits/tree");
  for ( int i=1; i<=2; ++i ) {
    myCanvas->cd(i);
    range->Draw();
    tree0->Draw("y:x","","same");
    //tree0->Draw("nphotons","");
    t->DrawLatex(-40,-10,"angle = 0");
  }
  
  // Draw other angles
  for ( int ifile = 1; ifile<nfiles; ++ifile ) {

    int ican = 2*ifile+1-(ifile/5)*7; // Sorry for that...
    std::cout << ican << std::endl;
    myCanvas->cd( ican );
    sprintf(fileName,"%s-%1d.root",fileRoot.Data(),ifile+1);
    TFile* file = new TFile(fileName);
    TTree* tree = file->Get("g4SimHits/tree");
    range->Draw();
    tree->Draw("y:x","","same");
//     tree->Draw("nphotons","");
    if ( ifile<5) sprintf(angle,"angle = %02d",(ifile*10));
    else sprintf(angle,"angle = %02d",-((ifile-4)*10));
    t->DrawLatex(-40,-10,angle);
  }


  return 0;

}
