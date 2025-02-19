///////////////////////////////////////////////////////////////////////////////
// File: HitDigitizerFP420.cc
// Date: 08.2008
// Description: HitDigitizerFP420 for FP420
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimRomanPot/SimFP420/interface/HitDigitizerFP420.h"
//#include "SimG4CMS/FP420/interface/FP420G4HitCollection.h"
//#include "SimG4CMS/FP420/interface/FP420G4Hit.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "SimRomanPot/SimFP420/interface/ChargeDrifterFP420.h"
#include "SimRomanPot/SimFP420/interface/ChargeDividerFP420.h"
#include "SimRomanPot/SimFP420/interface/InduceChargeFP420.h"

using namespace std;
#include<vector>

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

//HitDigitizerFP420::HitDigitizerFP420(float in,float inp,float inpx,float inpy){
//HitDigitizerFP420::HitDigitizerFP420(float in,float inp,float inpx,float inpy,float ild,float ildx,float ildy){
HitDigitizerFP420::HitDigitizerFP420(float in,float ild,float ildx,float ildy,float in0,float in2,float in3,int verbosity){
  moduleThickness =in; 
  double bz420 = in0; 
  double bzD2 =  in2; 
  double bzD3 =  in3;
  //  double pitch =inp; 
  //  double pitchX =inpx; 
  //  double pitchY =inpy; 
  double ldrift =ild; 
  double ldriftX =ildx; 
  double ldriftY =ildy; 
  //
  // Construct default classes
  //
  
  //theCDividerFP420 = new ChargeDividerFP420(pitch);
  theCDividerFP420 = new ChargeDividerFP420(moduleThickness, bz420, bzD2, bzD3, verbosity);
  
  depletionVoltage=20.0; //
  appliedVoltage=25.0;  //  a bit bigger than depletionVoltage to have positive value A for logA, A=1-2*Tfract.*Vd/(Vd+Vb)
  
  //  chargeMobility=480.0;//  = 480.0  !holes    mobility [cm**2/V/sec] p-side;   = 1350.0 !electron mobility - n-side
  chargeMobility=1350.0;//  = 480.0  !holes    mobility [cm**2/V/sec] p-side;   = 1350.0 !electron mobility - n-side
  //temperature=297.; // 24 C degree +273 = 297 ---->diffusion const for electrons= (1.38E-23/1.6E-19)*1350.0*297=34.6
  //  diffusion const for holes  =   12.3   [cm**2/sec]
  temperature=263.; // -10  C degree +273 = 263 ---->diffusion const for electrons= (1.38E-23/1.6E-19)*1350.0*263=30.6
  double diffusionConstant = CBOLTZ/e_SI * chargeMobility * temperature;
  // noDiffusion=true; // true if no Diffusion
  noDiffusion=false; // false if Diffusion
  if (noDiffusion) diffusionConstant *= 1.0e-3;
  
  chargeDistributionRMS=6.5e-10;
  
  // arbitrary:
  tanLorentzAnglePerTesla = 0. ;       //  try =0.106 if B field exist
  
  //    double timeNormalisation = pow(moduleThickness,2)/(2.*depletionVoltage*chargeMobility);
  //    double timeNormalisation = pow(pitch,2)/(2.*depletionVoltage*chargeMobility);
  
  double timeNormalisation = pow(ldrift,2)/(2.*depletionVoltage*chargeMobility);// 
  // double timeNormalisation = pow(ldrift,2)/(2.*depletionVoltage*chargeMobility);// 
  timeNormalisation = timeNormalisation*0.01; // because ldrift in [mm] but mu_e in [cm2/V/sec 
  
  //      double timeNormalisation = pow(pitch/2.,2)/(2.*depletionVoltage*chargeMobility);// i entered pitch as a distance between 2 read-out strips, not real distance between any (p or n) electrods which will be in 2 times less. But in expression for timeNormalisation the real distance of charge collection must be, so use pitch/2.   !
  
  
  double clusterWidth=5.;// was = 3
  gevperelectron= 3.61e-09; //     double GevPerElectron = 3.61e-09
  
  // GevPerElectron AZ:average deposited energy per e-h pair [keV]??? =0.0036
  if(verbosity>0) {
    std::cout << "HitDigitizerFP420: constructor ldrift= " << ldrift << std::endl;
    std::cout << "ldriftY= " <<ldriftY  << "ldriftX= " <<ldriftX  << std::endl;
    std::cout << "depletionVoltage" <<  depletionVoltage    << "appliedVoltage" <<  appliedVoltage    << "chargeMobility" <<  chargeMobility    << "temperature" <<  temperature    << "diffusionConstant" <<  diffusionConstant    << "chargeDistributionRMS" <<   chargeDistributionRMS   << "moduleThickness" <<   moduleThickness   << "timeNormalisation" <<  timeNormalisation    << "gevperelectron" <<  gevperelectron    << std::endl;
  }
  //ndif
  
  theCDrifterFP420 = 
    new ChargeDrifterFP420(moduleThickness,
			   timeNormalisation,
			   diffusionConstant,
			   temperature,
			   chargeDistributionRMS,
			   depletionVoltage,
			   appliedVoltage,
			   ldriftX,
			   ldriftY, verbosity);
  //					pitchX,
  //					pitchY);
  
  theIChargeFP420 = new InduceChargeFP420(clusterWidth,gevperelectron);
}


HitDigitizerFP420::~HitDigitizerFP420(){
  delete theCDividerFP420;
  delete theCDrifterFP420;
  delete theIChargeFP420;
}


//HitDigitizerFP420::hit_map_type HitDigitizerFP420::processHit(const PSimHit& hit, G4ThreeVector bfield, int xytype,int numStrips, double pitch){
HitDigitizerFP420::hit_map_type HitDigitizerFP420::processHit(const PSimHit& hit, G4ThreeVector bfield, int xytype,int numStrips, double pitch, int numStripsW, double pitchW, double moduleThickness, int verbosity){
  
  // use chargePosition just for cross-check in "induce" method
  // hit center in 3D-detector r.f.
  
  float middlex = (hit.exitPoint().x() + hit.entryPoint().x() )/2.;
  float middley = (hit.exitPoint().y() + hit.entryPoint().y() )/2.;
  
  
  float chargePosition= -100.;
  // Y: 
  if(xytype == 1) {
    //     chargePosition  = fabs(-numStrips/2. - ( int(middle.x()/pitch) +1.) );
    //chargePosition  = fabs(int(middle.x()/pitch+0.5*(numStrips+1)) + 1.);
    //      chargePosition  = fabs(int(middle.y()/pitch+0.5*(numStrips+1)) + 1.);
    // local and global reference frames are rotated in 90 degree, so global X and local Y are collinear
    //     chargePosition  = int(fabs(middle.x()/pitch + 0.5*numStrips + 1.));// charge in strip coord 
    chargePosition = 0.5*(numStrips) + middlex/pitch ;// charge in strip coord 0 - numStrips-1
    
    
  }
  // X:
  else if(xytype == 2) {
    //     chargePosition  = fabs(-numStrips/2. - ( int(middle.y()/pitch) +1.) );
    //chargePosition  = fabs(int(middle.y()/pitch+0.5*(numStrips+1)) + 1.);
    //      chargePosition  = fabs(int(middle.x()/pitch+0.5*(numStrips+1)) + 1.);
    // local and global reference frames are rotated in 90 degree, so global X and local Y are collinear
    //     chargePosition  = int(fabs(middle.y()/pitch + 0.5*numStrips + 1.));
    chargePosition = 0.5*(numStrips) + middley/pitch ;// charge in strip coord 0 - numStrips-1
    
    //  std::cout << " chargePosition    SiHitDi... = " << chargePosition                       << std::endl;
  }
  else {
    std::cout <<"================================================================"<<std::endl;
    std::cout << "****   HitDigitizerFP420:  !!!  ERROR: you have not to be here !!!  xytype=" << xytype << std::endl;
    // std::cout << "****   HitDigitizerFP420:  !!!  ERROR: you have not to be here !!!  xytype=" << xytype << std::endl;
    
    
    //     break;
  }
  //   if(chargePosition > numStrips || chargePosition<1) {
  if(chargePosition > numStrips || chargePosition < 0) {
    std::cout << "****   HitDigitizerFP420:  !!!  ERROR: check correspondence of XY detector dimensions in XML and here !!! chargePosition = " << chargePosition << std::endl;
    //     break;
  }
  
  if(verbosity>0) {
    std::cout << " ======   ****   HitDigitizerFP420:  !!!  processHit  !!!  : input: xytype=" << xytype << " numStrips=  " << numStrips << " pitch=  " << pitch << " Calculated chargePosition=  " << chargePosition << std::endl;
    std::cout << "The middle of hit point on input was: middlex =  " << middlex << std::endl;
    std::cout << "The middle of hit point on input was: middley =  " << middley << std::endl;
    //  std::cout << "For checks: hit point Entry =  " << hit.getEntry() << std::endl;
    std::cout << " ======   ****   HitDigitizerFP420:processHit:   start  CDrifterFP420 divide" << std::endl;
  }
  //
  // Fully process one SimHit
  //
  
  //   CDrifterFP420::ionization_type ion = theCDividerFP420->divide(hit,pitch);
  CDrifterFP420::ionization_type ion = theCDividerFP420->divide(hit,moduleThickness);
  //
  // Compute the drift direction for this det
  //
  
  //  G4ThreeVector driftDir = DriftDirection(&det,bfield);
  G4ThreeVector driftDir = DriftDirection(bfield,xytype,verbosity);
  if(verbosity>0) {
    std::cout << " ======   ****   HitDigitizerFP420:processHit: driftDir= " << driftDir << std::endl;
    std::cout << " ======   ****   HitDigitizerFP420:processHit:  start   induce , CDrifterFP420   drift   " << std::endl;
  }
  
  //  if(driftDir.z() ==0.) {
  //    std::cout << " pxlx: drift in z is zero " << std::endl; 
  //  }  else  
  //

  return theIChargeFP420->induce(theCDrifterFP420->drift(ion,driftDir,xytype), numStrips, pitch, numStripsW, pitchW, xytype, verbosity);
  
  //
}



G4ThreeVector HitDigitizerFP420::DriftDirection(G4ThreeVector _bfield, int xytype, int verbosity){
  
  // LOCAL hit: exchange xytype:  1 <-> 2
  
  //  Frame detFrame(_detp->surface().position(),_detp->surface().rotation());
  //  G4ThreeVector Bfield=detFrame.toLocal(_bfield);
  G4ThreeVector Bfield=_bfield;
  float dir_x, dir_y, dir_z; 
  // this lines with dir_... have to be specified(sign?) in dependence of field direction: 
  /*
    float dir_x = tanLorentzAnglePerTesla * Bfield.y();
    float dir_y = -tanLorentzAnglePerTesla * Bfield.x();
    float dir_z = 1.; // if E field is in z direction
  */
  // for electrons:
  // E field is either in X or Y direction with change vector to opposite
  
  // global Y or localX
  // E field is in Xlocal direction with change vector to opposite
  if(xytype == 2) {
    dir_x = 1.; // E field in Xlocal direction
    dir_y = +tanLorentzAnglePerTesla * Bfield.z();
    dir_z = -tanLorentzAnglePerTesla * Bfield.y();
  }
  // global X
  // E field is in Ylocal direction with change vector to opposite
  else if(xytype == 1) {
    dir_x = +tanLorentzAnglePerTesla * Bfield.z();
    dir_y = 1.; // E field in Ylocal direction
    dir_z = -tanLorentzAnglePerTesla * Bfield.x();
  }
  else{
    dir_x = 0.;
    dir_y = 0.;
    dir_z = 0.;
    std::cout << "HitDigitizerFP420: ERROR - wrong xytype=" <<  xytype   << std::endl;
  }
  
  
  //  G4ThreeVector theDriftDirection = LocalVector(dir_x,dir_y,dir_z);
  G4ThreeVector theDriftDirection(dir_x,dir_y,dir_z);
  // Local3DPoint EntryPo(aHit->getEntry().x(),aHit->getEntry().y(),aHit->getEntry().z());
  if(verbosity>0) {
    std::cout << "HitDigitizerFP420:DriftDirection tanLorentzAnglePerTesla= " << tanLorentzAnglePerTesla    << std::endl;
    std::cout << "HitDigitizerFP420:DriftDirection The drift direction in local coordinate is " <<  theDriftDirection    << std::endl;
  }
  
  return theDriftDirection;
  
}
