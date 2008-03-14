int plot( TString fileName = "Cherenkov.root" )
{
   TCanvas* myCanvas = new TCanvas("myCanvas","Canvas",10,10,600,900);
   myCanvas->Divide(1,2);

   TFile f(fileName);
   f.cd("analyzer");

   myCanvas->cd(1);
   hEnergy->Draw();
   myCanvas->cd(2);
   hTimeStructure->Draw();

   return 0;
}


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

int getAngleParams( double angle )
{

  double meanY = 1.45; // Y position of gun - (width crystal)/2

  double radAngle = angle*TMath::Pi()/180.;

  double meanX = meanY*TMath::Tan( radAngle );
  
  double phi = TMath::PiOver2()-radAngle;

  std::cout << "replace VtxSmeared.MeanX = " << meanX << std::endl;
  cout.precision(16);
  std::cout << "replace FlatRandomEGunSource.PGunParameters.MinPhi = " << phi << std::endl;
  std::cout << "replace FlatRandomEGunSource.PGunParameters.MaxPhi = " << phi << std::endl;

  return 0;

}
