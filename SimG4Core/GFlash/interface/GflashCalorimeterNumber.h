#ifndef SimG4Core_GFlash_GflashCalorimeterNumber_H
#define SimG4Core_GFlash_GflashCalorimeterNumber_H

enum GflashCalorimeterNumber { 
  kNULL,              
  kESPM,              // ECAL Barrel - ESPM
  kENCA,              // ECAL Endcap - ENCA
  kHB,                // HCAL Barrel - HB
  kHE,                // HCAL Endcap - HE
  kNumberCalorimeter
};

const G4String GflashCalorimeterName[kNumberCalorimeter] = {
  "NULL",
  "ESPM",
  "ENCA"
  "HB",
  "HE"
};

#endif
