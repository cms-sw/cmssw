//
//  SimRecStructures.h
//  
//
//  Created by Claudio Caputo on 13/02/14.
//
//

#ifndef _SimRecStructures_h
#define _SimRecStructures_h

struct MyGEMRecHit
{
    Int_t detId, particleType;
    Float_t x, y, xErr;
    Int_t region, ring, station, layer, chamber, roll;
    Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
    Int_t bx, clusterSize, firstClusterStrip;
    Float_t x_sim, y_sim;
    Float_t globalEta_sim, globalPhi_sim, globalX_sim, globalY_sim, globalZ_sim;
    Float_t pull;
};

struct MyGEMSimHit
{
    Int_t eventNumber;
    Int_t detUnitId, particleType;
    Float_t x, y, energyLoss, pabs, timeOfFlight;
    Int_t region, ring, station, layer, chamber, roll;
    Float_t globalR, globalEta, globalPhi, globalX, globalY, globalZ;
    Int_t strip;
    Float_t Phi_0, DeltaPhi, R_0;
    Int_t countMatching;
};

struct MySimTrack
{
    Float_t pt, eta, phi;
    Char_t charge;
    Char_t endcap;
    Char_t gem_sh_layer1, gem_sh_layer2;
    Char_t gem_rh_layer1, gem_rh_layer2;
    Float_t gem_sh_eta, gem_sh_phi;
    Float_t gem_sh_x, gem_sh_y;
    Float_t gem_rh_eta, gem_rh_phi;
    Float_t gem_lx_even, gem_ly_even;
    Float_t gem_lx_odd, gem_ly_odd;
    Char_t has_gem_sh_l1, has_gem_sh_l2;
    Char_t has_gem_rh_l1, has_gem_rh_l2;
    Float_t gem_trk_eta, gem_trk_phi, gem_trk_rho;
};

#endif
