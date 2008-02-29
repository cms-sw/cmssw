#include "SimG4Core/GFlash/interface/GflashHistogram.h"

GflashHistogram* GflashHistogram::instance_ = 0;

GflashHistogram* GflashHistogram::instance(){

  if(instance_ == 0) instance_ = new GflashHistogram();
  return instance_;

}

GflashHistogram::GflashHistogram() :
  theStoreFlag(false)
{
}

void GflashHistogram::bookHistogram(TString histFileName) 
{
  histFile_ = new TFile(histFileName,"RECREATE");

  TH1::AddDirectory(kTRUE);

  histFile_->mkdir("GflashEMShowerProfile");
  histFile_->cd("GflashEMShowerProfile");

  incE_atEcal = new TH1F("incE_atEcal","Incoming energy at Ecal;E (GeV);Number of electrons",50,0.0,50.0);
  dEdz        = new TH2F("dEdz","longitudinal profile;z (X_{0});E (GeV)",100,0.0,50.0,100,0.0,1.0);
  dEdz_p      = new TH1F("dEdz_p","longitudinal profile;z (X_{0});E (GeV)",100,0.0,50.0);
  dndz_spot   = new TH1F("dndz_spot","longitudinal profile;z (X_{0});Number of spots",30,0.0,30.0);
  rxry        = new TH2F("rxry","lateral profile;x (Moliere);y (Moliere)",100,-5.0,5.0,100,-5.0,5.0);
  dx          = new TH1F("dx","lateral profile;r (Moliere);Number of spots",200,-5.0,5.0);
  xdz         = new TH2F("xdz","Total shower profile;z (X_{0});r (Moliere)",30,0.0,30.0,200,-5.0,5.0);
  rzSpots     = new TH2F("rzSpots","r-z of spots in global coordinate;z (cm);r (cm)",1000,-500.0,500.0,400,-200.0,200.0);
  rho_ssp     = new TH1F("rho_ssp","Shower starting position;#rho (cm);Number of electrons",100,100.0,200.0);
  rArm        = new TH1F("rArm","lateral profile of abs radius;r (Moliere);Number of spots",100,0.0,5.0);

  histFile_->mkdir("GflashHadronShowerModel");
  histFile_->cd("GflashHadronShowerModel");

  preStepPosition  = new TH1F("preStepPosition","PreStep Position Shower",500,120.0,270.);
  postStepPosition = new TH1F("postStepPosition","PostStep Position Shower",500,120.0,270.);
  deltaStep        = new TH1F("deltaStep","Delta Step",200,0.0,100.);
  kineticEnergy    = new TH1F("kineticEnergy","Kinetic Energy",200,0.0,200.);
  energyLoss       = new TH1F("energyLoss","Energy Loss",200,0.0,200.);

  histFile_->mkdir("GflashHadronShowerProfile");
  histFile_->cd("GflashHadronShowerProfile");

  rshower = new TH1F("rshower","Lateral Lever" ,200,0.,100.);
  lateralx = new TH1F("lateralx","Lateral-X Distribution" ,200,-100.,100.);
  lateraly = new TH1F("lateraly","Lateral-Y Distribution" ,200,-100.,100.);

  gfhlong = new TH1F("gfhlong","Longitudinal Profile ",160,0.0,160.0);
  gfhsll  = new TH1F("gfhsll","StepLengthLeft ",200,0.0,200.0);
  gfhssp = new TH1F("gfhssp","Shower Starting Position",160,120.0,280.);
  gfheinc = new TH1F("gfheinc","Energy at Shower Starting Position",300,0.0,150.0);
  gfhlongProfile   = new TH2F("gfhlongProfile","Longitudinal Profile (all hits)",160,0.0,160.,60,125,245);
}


GflashHistogram::~GflashHistogram(){

  histFile_->cd();
  histFile_->Write();
  histFile_->Close();

}

