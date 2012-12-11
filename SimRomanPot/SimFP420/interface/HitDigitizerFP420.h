#ifndef HitDigitizerFP420_h
#define HitDigitizerFP420_h
 
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"
#include "SimRomanPot/SimFP420/interface/CDrifterFP420.h"
#include "SimRomanPot/SimFP420/interface/CDividerFP420.h"
#include "SimRomanPot/SimFP420/interface/IChargeFP420.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"


#include<vector>
#include <map>

// Digitizes the response for a single FP420Hit
class HitDigitizerFP420{
 public:

  typedef std::map<int, float, std::less<int> > hit_map_type;
  
  //HitDigitizerFP420(const edm::ParameterSet& conf, const ElectrodGeomDetUnit *det);
  //HitDigitizerFP420(float in, float inp, float inpx, float inpy,float ild,float ildx,float ildy);
  HitDigitizerFP420(float in, float ild, float ildx, float ildy, float in0, float in2, float in3, int verbosity);
  //HitDigitizerFP420(float in, float inp, float inpx, float inpy);
  
  ~HitDigitizerFP420();
  
  void setChargeDivider(CDividerFP420* cd){
    if (theCDividerFP420) delete theCDividerFP420;
    theCDividerFP420 = cd;
  }
  void setChargeCollectionDrifter(CDrifterFP420* cd){
    if (theCDrifterFP420) delete theCDrifterFP420;
    theCDrifterFP420 = cd;
  }
  void setInduceChargeOnElectrods(IChargeFP420* cd){
    if (theIChargeFP420) delete theIChargeFP420;
    theIChargeFP420 = cd;
  }
  
  //CDividerFP420* getChargeDivider(){return theCDividerFP420;}
  //CDrifterFP420* getChargeCollectionDrifter(){return theCDrifterFP420;}
  //IChargeFP420* getInduceChargeOnElectrods(){return theIChargeFP420;}
  
  //  hit_map_type processHit(const PSimHit&, G4ThreeVector, int, int, double);
  hit_map_type processHit(const PSimHit&, G4ThreeVector, int, int, double, int, double, double, int);
  
 private:
  CDividerFP420* theCDividerFP420;
  CDrifterFP420* theCDrifterFP420;
  IChargeFP420* theIChargeFP420;
  
  double moduleThickness;
  //  double pitch;
  // double pitchX;
  // double pitchY;
  double depletionVoltage;
  double appliedVoltage;
  double chargeMobility;
  double temperature;
  bool noDiffusion;
  double chargeDistributionRMS;
  double gevperelectron;
  
  G4ThreeVector DriftDirection(G4ThreeVector,int,int);
  
  //  typedef GloballyPositioned<double>      Frame;  //  AZ
  
  float tanLorentzAnglePerTesla;   //Lorentz angle tangent per Tesla
  
};

#endif
