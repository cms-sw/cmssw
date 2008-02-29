#ifndef GflashNameSpace_H
#define GflashNameSpace_H

#include "globals.hh"

namespace Gflash {

  //   enum

  enum CalorimeterNumber {
    kNULL,
    kESPM,              // ECAL Barrel - ESPM
    kENCA,              // ECAL Endcap - ENCA
    kHB,                // HCAL Barrel - HB
    kHE,                // HCAL Endcap - HE
    kNumberCalorimeter
  };
  
  const G4String CalorimeterName[kNumberCalorimeter] = {
    "NULL",
    "ESPM",
    "ENCA"
    "HB",
    "HE"
  };

  const G4double Zmin[kNumberCalorimeter] = {0.0,0.000,304.5,0.000,391.95};
  const G4double Zmax[kNumberCalorimeter] = {0.0,317.0,390.0,433.2,554.10};
  const G4double Rmin[kNumberCalorimeter] = {0.0,123.8, 31.6,177.5,31.6};
  const G4double Rmax[kNumberCalorimeter] = {0.0,175.0,171.1,287.7,263.90};
  const G4double EtaMin[kNumberCalorimeter] = {0.00,0.00,1.57,0.00,1.57};
  const G4double EtaMax[kNumberCalorimeter] = {0.00,1.30,3.00,1.30,3.00};
    
  //constants needed for GflashHadronShowerProfile

  // number of sub-detectors (calorimeters)  
  const G4int NDET = 4; 
  
  const G4double FLUHAD[3][NDET] = {{0.16,.161,0.150,0.130},
				    {0.,0.,0.,0.},
				    {0.044,0.044,0.053,0.040}};
  const G4double SAMHAD[3][NDET] = {{0.12,0.35,0.18,0.23},
				    {0.,0.,0.,0.},
				    {0.010,0.032,0.038,0.043}};
  const G4double RLTHAD[NDET] = {32.7,23.7,32.7,23.7};
  
  const G4double PBYMIP[NDET] = {1.82,3.20,1.85,2.3};

  //utility functions

}

#endif
