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
  const G4double Zmin[kNumberCalorimeter]   = {0.0, 0.000, 304.5, 0.000, 391.95}; // in cm
  const G4double Zmax[kNumberCalorimeter]   = {0.0, 317.0, 390.0, 433.2, 554.10};
  const G4double Rmin[kNumberCalorimeter]   = {0.0, 123.8,  31.6, 177.5,  31.6};
  const G4double Rmax[kNumberCalorimeter]   = {0.0, 175.0, 171.1, 287.7, 263.9};
  const G4double EtaMin[kNumberCalorimeter] = {0.0, 0.000, 1.570, 0.000, 1.570};
  const G4double EtaMax[kNumberCalorimeter] = {0.0, 1.300, 3.000, 1.300, 3.000};
    
  //constants needed for GflashHadronShowerProfile

  const G4double rMoliere[kNumberCalorimeter]  = {2.19, 2.19, 2.19, 2.19, 2.19}; // in cm
  const G4double radLength[kNumberCalorimeter] = {0.89, 0.89, 0.89, 16.42, 16.42}; // in cm
  const G4double Z[kNumberCalorimeter]         = {68.360, 68.360, 68.360, 68.360, 68.360}; // mass of molicule
  const G4double criticalEnergy                = 8.6155 / GeV;

  //@@@approximately ScaleSensitive = 0.2 and need fine tune later 
  //@@@set it to 1.0 for the energy shape studies
  const G4double ScaleSensitive = 0.26;

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

  // correlation matrix RHO[I][J]
  const G4int NRegion   = 3;
  const G4int NxN  = 6;
  const int NDim[NRegion] = {6,6,6};
  const G4int NStart[NRegion] = {0,21,42};

  const G4double rho[NRegion*NxN][NxN] = {
    { 1.}, 
    {-0.532, 1.},
    {-0.330, 0.581, 1.},
    {-0.026, 0.289, 0.192, 1.},
    { 0.001,-0.004,-0.009, 0.028, 1.},
    {-0.062,-0.031,-0.022,-0.019,-0.434, 1.},
    { 1.}, 
    {-0.593, 1.},
    {-0.277, 0.492, 1.},
    {-0.201, 0.443, 0.100, 1.},
    { 0.017,-0.102, 0.088,-0.079, 1.},
    {-0.157, 0.112, 0.031,-0.018, 0.230, 1.},
    { 1.}, 
    {-0.593, 1.},
    {-0.277, 0.492, 1.},
    {-0.201, 0.443, 0.100, 1.},
    { 0.017,-0.102, 0.088,-0.079, 1.},
    {-0.157, 0.112, 0.031,-0.018, 0.230, 1.}};
}

#endif
