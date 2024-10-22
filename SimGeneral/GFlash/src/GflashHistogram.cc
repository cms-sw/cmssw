#include "SimGeneral/GFlash/interface/GflashHistogram.h"

GflashHistogram *GflashHistogram::instance_ = nullptr;

GflashHistogram *GflashHistogram::instance() {
  if (instance_ == nullptr)
    instance_ = new GflashHistogram();
  return instance_;
}

GflashHistogram::GflashHistogram() : theStoreFlag(false) {}

void GflashHistogram::bookHistogram(std::string histFileName) {
  histFile_ = new TFile(histFileName.c_str(), "RECREATE");

  TH1::AddDirectory(kTRUE);

  histFile_->mkdir("GflashEMShowerProfile");
  histFile_->cd("GflashEMShowerProfile");

  em_incE = new TH1F("em_incE", "Incoming energy at Ecal;E (GeV);Number of Events", 500, 0.0, 500.0);
  em_ssp_rho = new TH1F("em_ssp_rho", "Shower starting position;#rho (cm);Number of Events", 100, 100.0, 200.0);
  em_ssp_z = new TH1F("em_ssp_z", "Shower starting position;z (cm);Number of Events", 800, -400.0, 400.0);
  em_long = new TH1F("em_long", "Longitudinal Profile;Radiation Length;Number of Spots", 100, 0.0, 50.0);
  em_lateral = new TH2F(
      "em_lateral", "Lateral Profile vs. Shower Depth;Radiation Length;Moliere Radius", 100, 0.0, 50.0, 100, 0.0, 3.0);
  em_long_sd = new TH1F("em_long_sd",
                        "Longitudinal Profile in Sensitive Detector;Radiation "
                        "Length;Number of Spots",
                        100,
                        0.0,
                        50.0);
  em_lateral_sd = new TH2F("em_lateral_sd",
                           "Lateral Profile vs. Shower Depth in Sensitive "
                           "Detector;Radiation Length;Moliere Radius",
                           100,
                           0.0,
                           50.0,
                           100,
                           0.0,
                           3.0);
  em_nSpots_sd = new TH1F("em_nSpots_sd",
                          "Number of Gflash Spots in Sensitive Detector;Number "
                          "of Spots;Number of Events",
                          1000,
                          0.0,
                          100000);

  histFile_->mkdir("GflashHadronShowerModel");
  histFile_->cd("GflashHadronShowerModel");

  preStepPosition = new TH1F("preStepPosition", "PreStep Position Shower", 500, 120.0, 270.);
  postStepPosition = new TH1F("postStepPosition", "PostStep Position Shower", 500, 120.0, 270.);
  deltaStep = new TH1F("deltaStep", "Delta Step", 200, 0.0, 100.);
  kineticEnergy = new TH1F("kineticEnergy", "Kinetic Energy", 200, 0.0, 200.);
  energyLoss = new TH1F("energyLoss", "Energy Loss", 200, 0.0, 200.);
  energyRatio = new TH1F("energyRatio", "energyLeading/energyTotal", 200, 0.0, 1.);

  histFile_->mkdir("GflashHadronShowerProfile");
  histFile_->cd("GflashHadronShowerProfile");

  rshower = new TH1F("rshower", "Lateral Lever", 200, 0., 100.);
  lateralx = new TH1F("lateralx", "Lateral-X Distribution", 200, -100., 100.);
  lateraly = new TH1F("lateraly", "Lateral-Y Distribution", 200, -100., 100.);
  gfhlongProfile = new TH2F("gfhlongProfile", "Longitudinal Profile (all hits)", 160, 0.0, 160., 60, 125, 245);

  histFile_->mkdir("GflashWatcher");
  histFile_->cd("GflashWatcher");

  g4vertexTrack = new TH1F("g4vertexTrack", "Vertex of Track", 300, 0.0, 300.);
  g4stepCharge = new TH2F("g4stepCharge", "Geant4 Step Charge", 300, 120., 420, 5, -2.5, 2.5);
  g4nSecondary = new TH1F("g4nSecondary", "number of Secondaries", 100, 0.0, 100.);
  g4pidSecondary = new TH1F("g4pidSecondary", "PID of Secondaries", 3000, -500.0, 2500.);

  g4energySecondary = new TH1F("g4energySecondary", "Kinetic Energy of Secondaries", 300, 0.0, 15.);
  g4energyPi0 = new TH1F("g4energyPi0", "Kinetic Energy of Pi0", 300, 0.0, 15.);
  g4energyElectron = new TH1F("g4energyElectron", "Kinetic Energy of Electron", 300, 0.0, 15.);
  g4energyPhoton = new TH1F("g4energyPhoton", "Kinetic Energy of Photon", 300, 0.0, 15.);

  g4totalEnergySecPhoton = new TH1F("g4totalEnergySecPhoton", "Total Kinetic Energy of Sec Photon", 300, 0.0, 3.);
  g4totalEnergySecElectron = new TH1F("g4toalEnergySecElectron", "Total Kinetic Energy of Sec Electron", 300, 0.0, 3.);
  g4totalEnergySecPi0 = new TH1F("g4totalEnergySecPi0", "Total Kinetic Energy of Sec Pi0", 300, 0.0, 30.);

  g4energyEM = new TH1F("energyEM", "EM Energy", 600, 0.0, 150.0);
  g4energyHad = new TH1F("energyHad", "Had Energy", 600, 0.0, 150.0);
  g4energyTotal = new TH1F("energyTotal", "Total Energy", 600, 0.0, 150.0);
  g4energyEMMip = new TH1F("energyEMMip", "EM Energy (MIP)", 100, 0.0, 2.0);
  g4energyHadMip = new TH1F("energyHadMip", "Had Energy (MIP)", 600, 0.0, 150.0);
  g4energyMip = new TH1F("energyMip", "Total Energy (MIP)", 600, 0.0, 150.0);
  g4energyEMvsHad = new TH2F("energyEMvsHad", "EM Energy", 600, 0.0, 150.0, 600, 0.0, 150.0);

  g4energySensitiveEM = new TH1F("energySensitiveEM", "Sensitive EM Energy", 600, 0.0, 150.0);
  g4energySensitiveHad = new TH1F("energySensitiveHad", "Sensitive Had Energy", 600, 0.0, 150.0);
  g4energySensitiveTotal = new TH1F("energySensitiveTotal", "Sensitive Total Energy", 600, 0.0, 150.0);
  g4energyHybridTotal = new TH1F("energyHybridTotal", "Hybrid Total Energy", 600, 0.0, 150.0);
  g4energySensitiveEMMip = new TH1F("energySensitiveEMMip", "Sensitive EM Energy (MIP)", 100, 0.0, 2.0);
  g4energySensitiveEMvsHad =
      new TH2F("energySensitiveEMvsHad", "Sensitive EM Energy vs Had", 600, 0.0, 150.0, 600, 0.0, 150.0);

  g4energyEMProfile = new TH2F("energyEMProfile", "EM Energy Profile", 600, 0.0, 150.0, 60, 125, 245);
  g4energyHadProfile = new TH2F("energyHadProfile", "Had Energy Profile", 600, 0.0, 150.0, 60, 125, 245);
  g4energyTotalProfile = new TH2F("energyTotalProfile", "Total Energy Profile", 600, 0.0, 150.0, 60, 125, 245);
  g4energyHybridProfile = new TH2F("energyHybridProfile", "Hybrid Energy Profile", 600, 0.0, 150.0, 60, 125, 245);

  g4ssp = new TH1F("g4ssp", "Shower Starting Position", 160, 120.0, 280.);
  g4energy = new TH1F("g4energy", "Energy at Shower Starting Position", 600, 0.0, 150.0);
  g4energyLoss = new TH1F("g4energyLoss", "Energy Loss", 600, 0.0, 150.);
  g4momentum = new TH1F("g4momentum", "Momentum/GeV at Shower Starting Position", 300, 0.0, 150.0);
  g4charge = new TH1F("g4charge", "Track Charge at Shower Starting Position", 10, -5.0, 5.0);

  g4rshower = new TH2F("g4rshower", "rshower", 200, 0., 40, 25, 125., 250);
  g4rshowerR1 = new TH2F("g4rshowerR1", "rshower vs depth ssp=127-129", 200, 0., 40, 40, 0.0, 160.);
  g4rshowerR2 = new TH2F("g4rshowerR2", "rshower vs depth ssp=131-133", 200, 0., 40, 40, 0.0, 160.);
  g4rshowerR3 = new TH2F("g4rshowerR3", "rshower vs depth ssp=173-175", 200, 0., 40, 40, 0.0, 160.);

  g4lateralXY = new TH2F("g4lateralXY", "Lateral Profile XY", 160, -40., 40, 25, 125., 250);
  g4lateralRZ = new TH2F("g4lateralRZ", "Lateral Profile RZ", 160, -40., 40, 25, 125., 250);
  g4spotXY = new TH2F("g4spotXY", "x-y of spots in global coordinate", 800, -400., 400., 800, -400.0, 400.0);
  g4spotRZ = new TH2F("g4spotRZ", "r-z of spots in global coordinate", 1200, -1500., 1500., 400, .0, 400.0);
  g4spotRZ0 = new TH2F("g4spotRZ0", "all r-z of spots in global coordinate", 1200, -1500., 1500., 400, .0, 400.0);

  g4stepRho = new TH1F("g4stepRho", "rho of Geant4 Step", 200, 120., 320.);
  g4trajectoryPhi0 = new TH1F("g4trajectoryPhi0", "trajectory Phi0", 2000, -5., 5.);

  g4trajectoryXY = new TH2F("g4trajectoryXY", "trajectory x-y", 800, -400.0, 400.0, 800, -400.0, 400.0);
  g4trajectoryRZ = new TH2F("g4trajectoryRZ", "trajectory r-z ", 1200, -600.0, 600.0, 400, 0.0, 400.0);

  g4longProfile = new TH2F("g4longProfile", "Longitudinal Profile (all hits)", 160, 0.0, 160., 60, 125, 245);
  g4longDetector =
      new TH2F("g4longDetector", "Longitudinal Profile (hits inisde detectors)", 160, 0.0, 160., 60, 125, 245);
  g4longSensitive = new TH2F("g4longSensitive", "Longitudinal Profile (Sensitive)", 160, 0.0, 160., 60, 125, 245);
}

GflashHistogram::~GflashHistogram() {
  histFile_->cd();
  histFile_->Write();
  histFile_->Close();
}
