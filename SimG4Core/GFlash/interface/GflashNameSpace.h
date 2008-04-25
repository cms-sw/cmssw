#ifndef GflashNameSpace_H
#define GflashNameSpace_H

#include "globals.hh"

namespace Gflash {

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

  //                                                 EB     EE     HB     HE
  const G4double Zmin[kNumberCalorimeter]   = {0.0, 0.000, 304.5, 0.000, 391.95};
  const G4double Zmax[kNumberCalorimeter]   = {0.0, 317.0, 390.0, 433.2, 554.10};
  const G4double Rmin[kNumberCalorimeter]   = {0.0, 123.8,  31.6, 177.5,  31.6};
  const G4double Rmax[kNumberCalorimeter]   = {0.0, 175.0, 171.1, 287.7, 263.9};
  const G4double EtaMin[kNumberCalorimeter] = {0.0, 0.000, 1.570, 0.000, 1.570};
  const G4double EtaMax[kNumberCalorimeter] = {0.0, 1.300, 3.000, 1.300, 3.000};
    
  //constants needed for GflashHadronShowerProfile

  //@@@approximately ScaleSensitive = 0.2 and need fine tune later
  //temporarily we set it 1.0 for the energy shape studies
  const G4double ScaleSensitive = 1.0;

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
