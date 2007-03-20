#ifndef ChargeDividerFP420_h
#define ChargeDividerFP420_h

#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"

#include "SimRomanPot/SimFP420/interface/CDividerFP420.h"
#include "SimRomanPot/SimFP420/interface/LandauFP420.h"
#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
//#define mydigidebug11

class ChargeDividerFP420 : public CDividerFP420{
 public:

  ChargeDividerFP420(double pit){
#ifdef mydigidebug11
cout << "ChargeDividerFP420.h: constructor" << endl;
cout << "peakMode = " << peakMode << "fluctuateCharge=   "<< fluctuateCharge <<  "chargedivisionsPerHit = "  << chargedivisionsPerHit << "deltaCut=   "<< deltaCut << endl;
#endif
    FP420NumberingScheme * theFP420NumberingScheme;
  // Initialization:
  theFP420NumberingScheme = new FP420NumberingScheme();


  // Run APV in peak instead of deconvolution mode, which degrades the time resolution
//  peakMode=true ; //     APVpeakmode
  peakMode=false; //     APVconvolutionmode
  
  // Enable interstrip Landau fluctuations within a cluster.
  fluctuateCharge=true;   
  
  // Number of segments per strip into which charge is divided during simulation.
  // If large the precision of simulation improves.
  chargedivisionsPerHit=10; // = or =20
 
  // delta cutoff in MeV, has to be same as in OSCAR (0.120425 MeV corresponding // to 100um range for electrons)
  //SimpleConfigurable<double>  ChargeDividerFP420::deltaCut(0.120425,
  deltaCut=0.120425;  //  DeltaProductionCut

  pitchcur= pit;

  // but position before Stations:
        

  double zD2 = 1000.;  // dist between centers of 1st and 2nd stations
  double zD3 = 8000.;  // dist between centers of 1st and 3rd stations
  zStationBegPos[0] = -150. - (118.4+10.)/2; // 10. -arbitrary
  zStationBegPos[1] = zStationBegPos[0]+zD2;
  zStationBegPos[2] = zStationBegPos[0]+zD3;
  zStationBegPos[3] = zStationBegPos[0]+2*zD3;

}



  virtual ~ChargeDividerFP420(){
    //    delete particleTable;
  }

  //  CDividerFP420::ionization_type divide(const SimHit&, const StripDet& det);
  CDividerFP420::ionization_type divide(const FP420G4Hit&, const double&);
  
 private:

  //  double pit;
  double pitchcur;
  
  float PeakShape(const FP420G4Hit&);
  float DeconvolutionShape( const FP420G4Hit&);
  float TimeResponse( const FP420G4Hit&) ; 
  void fluctuateEloss(int particleId, float momentum, float eloss, float length, int NumberOfSegmentation, float elossVector[]);   
  //  static SimpleConfigurable<bool> peakMode;
  bool peakMode;
  //  static SimpleConfigurable<bool> fluctuateCharge;
  bool fluctuateCharge;
  //  static SimpleConfigurable<int>  chargeDivisionsPerStrip;
  int  chargedivisionsPerHit;
  float zStationBegPos[4];
  //  static SimpleConfigurable<double> deltaCut ;
  double deltaCut ;
  LandauFP420 fluctuate; 
  //  HepPDTable * particleTable;
};

#endif
