#ifndef ChargeDividerFP420_h
#define ChargeDividerFP420_h

#include "SimG4CMS/FP420/interface/FP420NumberingScheme.h"

#include "SimRomanPot/SimFP420/interface/CDividerFP420.h"
#include "SimRomanPot/SimFP420/interface/LandauFP420.h"
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class ChargeDividerFP420 : public CDividerFP420{
 public:
  
  explicit ChargeDividerFP420(double pit, double az420, double azD2, double azD3, int);
  
  
  virtual ~ChargeDividerFP420();
  
  //  CDividerFP420::ionization_type divide(const SimHit&, const StripDet& det);
  CDividerFP420::ionization_type divide(const PSimHit&, const double&);
  
 private:
  
  FP420NumberingScheme * theFP420NumberingScheme;

  double pitchcur; // is really moduleThickness here !!!
  double z420;  // dist between centers of 1st and 2nd stations
  double zD2;  // dist between centers of 1st and 2nd stations
  double zD3;  // dist between centers of 1st and 3rd stations
  
  float PeakShape(const PSimHit&);
  float DeconvolutionShape( const PSimHit&);
  float TimeResponse( const PSimHit&) ; 
  void fluctuateEloss(int particleId, float momentum, float eloss, float length, int NumberOfSegmentation, float elossVector[]);   
  //  static SimpleConfigurable<bool> peakMode;
  bool peakMode;
  bool decoMode;
  //  static SimpleConfigurable<bool> fluctuateCharge;
  bool fluctuateCharge;
  //  static SimpleConfigurable<int>  chargeDivisionsPerStrip;
  int  chargedivisionsPerHit;
  float zStationBegPos[4];
  //  static SimpleConfigurable<double> deltaCut ;
  double deltaCut ;
  LandauFP420 fluctuate; 
  //  HepPDTable * particleTable;
  int  verbosity;
};

#endif
