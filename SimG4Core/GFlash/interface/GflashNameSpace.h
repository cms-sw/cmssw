#ifndef GflashNameSpace_H
#define GflashNameSpace_H 

#include "globals.hh"
#include "G4ThreeVector.hh"

namespace Gflash {

  enum CalorimeterNumber {
    kNULL = -1,         // ECAL Barrel - ESPM
    kESPM,              // ECAL Barrel - ESPM
    kHB,                // HCAL Barrel - HB
    kENCA,              // ECAL Endcap - ENCA
    kHE,                // HCAL Endcap - HE
    kHF,                // HCAL Forward - HF
    kNumberCalorimeter
  };
  
  CalorimeterNumber getCalorimeterNumber(const G4ThreeVector position);

  //                                            EB     HB     EE     HE      HF
  const G4double Zmin[kNumberCalorimeter]   = {0.000, 0.000, 304.5, 391.95, 1110.0}; // in cm
  const G4double Zmax[kNumberCalorimeter]   = {317.0, 433.2, 390.0, 554.10, 1275.0};
  const G4double Rmin[kNumberCalorimeter]   = {123.8, 177.5,  31.6,  31.6,    12.5};
  const G4double Rmax[kNumberCalorimeter]   = {177.5, 287.7, 171.1, 263.9,   140.0};

  const G4double EtaMin[kNumberCalorimeter] = {0.000, 0.000, 1.570,  1.570, 3.000};
  const G4double EtaMax[kNumberCalorimeter] = {1.300, 1.300, 3.000,  3.000, 5.000};

  //constants needed for GflashHadronShowerModel
  const G4double energyCutOff                 = 1.0*GeV;
    
  //constants needed for GflashHadronShowerProfile

  const G4double rMoliere[kNumberCalorimeter]  = { 2.19, 2.19, 2.19, 2.19, 1.72}; // in cm
  const G4double radLength[kNumberCalorimeter] = { 0.89, 1.49, 0.89, 1.49, 1.76}; // in cm
  const G4double intLength[kNumberCalorimeter] = { 22.4,16.42, 22.4, 16.42, 16.77}; // in cm
  const G4double Z[kNumberCalorimeter]         = { 68.36, 68.36, 68.36, 68.36, 55.845}; // mass of molicule
  const G4double criticalEnergy                = 8.6155 / GeV;

  //cut value for quasi-elastic like interactions on the secondary energy, 
  //ratio = (Leading Energy of Secondaries)/(Total Energy)
  const G4double QuasiElasticLike = 0.95;

  //minimum distance to the back of parameterized envelopes in [cm]
  const G4double MinDistanceToOut = 10.0;

  //additional energy scale for the Hcal sensitive detector
  const G4double ScaleSensitive = 0.167;

  // properties for different sub-detectors (calorimeters)
  const G4double SAMHAD[3][kNumberCalorimeter] = {{0.0,0.89,0.0,0.89,0.0},
                                                  {0.0,0.00,0.0,0.00,0.0},
                                                  {0.0,0.00,0.0,0.00,0.0}};
  const G4double RLTHAD[kNumberCalorimeter] = {32.7,23.7,32.7,23.7,23.7};

  //parameters for logitudinal parameterizations and energy responses

  const G4int NPar  = 5;

  const G4double fdep[2][4] = {
    {  8.5073e-01,  2.9650e-02,  8.1327e-01,  2.7547e+00},
    {  9.7806e-02,  4.6989e-02,  0.0000e+00,  0.0000e+00}
  };

  const G4double emscale[4][4] = {
    {  5.4463e-01, -4.1210e-02,  1.8231e+00,  4.1472e+00 },
    { -3.4608e-01, -1.7809e-01,  1.1329e+00,  2.0367e+00 },
    {  1.7327e-01,  3.1683e-02,  1.0484e+00,  2.3877e+00 },
    { -1.2818e-01, -1.4000e-01,  2.9789e-01,  3.9971e+00 }
  };

  const G4double hadscale[7][4] = {
    //based on LogNormal of Had Energy + (0.05*Einc)% swift 
    { -3.1358e+01,  5.3531e+01,  3.8817e-02, -1.5548e+01 },
    {  8.2970e-01,  2.6359e-01, -1.2500e+00,  3.7566e+00 },
    {  6.1175e-01,  1.3195e-01, -2.0953e+00,  2.9990e+00 },
    { -3.1528e-01,  8.0146e-02, -6.1015e+00,  1.8951e+00 },
    //LogNormal of Had Energy for mip-like response based on 2006 H2 test beam data
    {  4.4527e+00,  5.7913e+00,  1.6584e-01,  4.6872e+00 },
    {  2.4081e-01,  1.6057e-01, -6.1439e-01,  1.9368e+00 },
    //correction of response if ssp is between Ecal and Hcal
    {  1.8736e-01,  1.7391e-01,  1.0991e+00,  3.3557e+00 }
  };

  const G4double protonscale[3][4] = {
    {  3.7385e+00,  3.3977e+00,  3.0798e-01,  3.8725e+00 },
    {  2.4081e-01,  1.6057e-01, -6.1439e-01,  1.9368e+00 },
    //correction of response if ssp is between Ecal and Hcal
    {  1.8736e-01,  1.7391e-01,  1.0991e+00,  3.3557e+00 }
  };


  const G4double par[8*NPar][4] = {
    {  6.7438e-01,  1.7413e-01,  8.1317e-01,  2.6713e+00 }, //Mean Ecal ShowerType1
    {  9.5916e-01,  1.0649e+00,  1.8075e-01,  3.0428e+00 },
    { -1.2427e+00,  2.9128e-01,  6.2858e-01,  2.6237e+00 },
    {  7.9194e-01, -4.9592e-01, -1.1214e+00,  2.6263e+00 },
    {  1.8503e+00,  1.7336e-01,  1.6952e+00,  2.9239e+00 },

    {  6.8778e-01,  1.3956e-01, -1.7557e+00,  1.5919e+00 }, //Mean Hcal ShowerType1
    {  7.0000e-01, -5.0000e-02,  4.2028e+00,  1.4877e+00 },
    { -1.5000e+00, -9.5593e-01,  2.3739e-01,  5.8041e-01 },
    {  3.0602e+00, -2.3879e+00,  4.6035e-01, -4.5823e-01 },
    {  2.0000e+00,  2.0000e+00, -4.0535e-01,  8.4006e-01 },

    {  6.0622e-01,  3.4491e-02,  1.9601e+00,  4.0597e+00 }, //Mean Hcal ShowerType2
    {  6.1856e-01, -5.3972e-02,  4.2128e+00,  3.0421e+00 },
    { -1.9157e+00,  5.9675e-01, -3.1110e-01,  2.3064e+00 },
    {  1.1704e+00,  4.1968e-01, -7.8091e-01,  1.6688e+00 },
    {  2.0000e+00, -2.0000e+00,  4.3481e-01,  7.7922e-01 },

    {  6.3159e-01,  1.4557e-01,  4.7778e-01,  2.5534e+00 }, //Mean Hcal ShowerType3
    {  6.4682e-01, -7.3109e-02,  3.4524e+01,  2.9954e+00 },
    { -1.9768e+00,  3.1293e-01, -5.6383e-01,  3.0850e+00 },
    {  9.6181e-01,  1.1237e-01, -1.7285e+00,  2.3366e+00 },
    {  1.5000e+00, -1.5000e+00,  4.2321e-01,  8.9595e-01 },

    { -4.4895e+00,  5.2126e+00, -2.3116e-02,  6.4173e+01 }, //Sigma Ecal ShowerType1
    {  3.3856e-01,  7.4435e-02, -1.4309e+00,  2.6740e+00 },
    {  2.6759e-01,  2.5205e-02, -2.3270e+00,  3.3109e+00 },
    {  6.7557e-01,  3.2703e-01, -7.6511e-01,  1.7957e+00 },
    {  4.3271e-01,  2.0000e-01, -4.7512e-01,  1.0227e+00 },

    {  3.0677e-01,  2.2520e-01, -6.2077e-01,  1.1081e+00 }, //Sigma Hcal ShowerType1
    {  4.0648e-01,  7.7278e-03,  2.4117e+02,  2.2477e+00 },
    {  4.4076e-01,  2.4788e-01, -2.5523e-01,  3.3124e+00 },
    {  1.1130e+00,  9.1704e-01, -3.6457e-01,  1.1608e+00 },
    {  1.5000e+00, -1.3000e+00,  3.2784e-01,  3.5213e-01 },

    {  2.5000e-01,  1.6444e-01, -3.9152e-01,  8.4963e-01 }, //Sigma Hcal ShowerType2
    {  4.9724e-01, -3.6006e-02,  1.1210e+00,  3.9498e+00 },
    {  4.2496e-01, -1.6087e-01,  4.0393e-01,  3.9751e+00 },
    {  8.5041e-01,  5.1464e-01, -6.6302e-01,  1.9777e+00 },
    {  1.3960e+00, -1.2054e+00,  3.3610e-01,  7.0376e-01 },

    {  2.5000e-01,  2.1781e-01, -2.2974e-01,  6.2906e-01 }, //Sigma Hcal ShowerType3
    {  5.2411e-01, -1.0607e-01,  7.4782e-01,  2.7602e+00 },
    {  1.0462e+00, -1.9208e+01,  2.7239e-03, -6.7807e+00 },
    {  6.6619e-01,  3.1562e-01, -7.8957e-01,  2.2168e+00 },
    {  8.2528e-01, -6.2222e-01,  3.9534e-01,  1.8290e+00 }
  };

  const G4double rho[8*NPar][4] = {
    {  1.4190e-01,  2.6308e-01,  1.2979e+00,  1.9024e+00 }, //Ecal ShowerType1
    {  4.7805e-01,  1.3546e-01,  1.5815e+00,  1.6697e+00 },
    {  8.2088e-01,  2.1833e-02,  2.2896e+00,  1.5505e+00 },
    { -5.2167e-02,  0.0000e+00,  1.0000e+00,  1.0000e+00 },
    {  2.7072e-01, -6.9839e-02,  1.0111e+02,  2.2077e+00 },
    {  1.3216e-01, -1.0180e-01,  9.8382e+00,  2.2553e+00 },
    {  5.3132e-02, -6.8272e-02,  1.3674e+01,  2.3076e+00 },
    {  2.7417e-01,  0.0000e+00,  4.0000e+00,  1.0000e+00 },
    {  2.1588e-01, -1.0111e-01,  1.1345e+01,  2.3139e+00 },
    {  8.0280e-01,  1.4780e-01,  8.5541e-01,  2.2995e+00 },

    {  5.9673e-02,  2.6255e-01,  7.3541e-01,  2.7660e+00 }, //Hcal ShowerType1
    {  2.7126e-01,  1.5810e-01,  1.3596e+00,  2.4690e+00 },
    {  7.5047e-01,  6.8428e-02,  8.7936e-01,  5.0565e+00 },
    {  8.9725e-02,  8.0812e-02,  2.2949e+00,  4.0534e+00 },
    {  5.0000e-01,  6.6307e+01,  1.6838e-03,  3.2369e+00 },
    {  2.7641e-01,  1.5000e-01,  1.0194e+00,  1.2387e+00 },
    {  1.2386e-01,  0.0000e+00,  1.0000e+00,  1.0000e+00 },
    {  3.8881e-01,  1.8000e-01,  8.3829e-01,  5.1864e+00 },
    {  4.6535e-01,  1.5000e-01,  9.3439e-01,  9.1905e-01 },
    {  8.0000e-01,  2.0000e-01, -5.7854e-01,  2.0167e+00 },

    { -2.2610e-02,  8.7079e-02,  1.3255e+01,  2.2260e+00 }, //Hcal ShowerType2
    {  2.7147e-01,  1.2332e-01,  3.2099e+00,  1.9717e+00 },
    {  6.6040e-01, -1.1842e-02,  6.8650e+00,  2.0085e+00 },
    {  1.0205e-01,  1.0000e-01,  5.4438e-01,  3.6551e+00 },
    {  3.0000e-01,  2.0000e-01,  8.2230e-01,  1.3549e+00 },
    {  3.4682e-01,  0.0000e+00,  1.0000e+00,  1.0000e+00 },
    {  1.7160e-01,  3.9781e-02,  6.7383e+00,  1.9256e+00 },
    {  2.0000e-01,  1.3237e-01,  2.5413e-01,  1.2529e+00 },
    {  6.1090e-01,  0.0000e+00,  1.0000e+00,  1.0000e+00 },
    {  7.2769e-01, -1.2600e-01,  8.1386e-01,  2.4818e+00 },

    {  1.0345e-01,  4.8906e-02,  6.6775e+01,  2.0782e+00 }, //Hcal ShowerType3
    {  4.2870e-01,  1.0980e-01,  4.7327e+00,  1.7190e+00 },
    {  7.0416e-01,  0.0000e+00,  1.0000e+00,  1.0000e+00 },
    { -1.0516e-02, -1.0000e-01,  1.0603e+00,  4.5948e+00 },
    {  3.9288e-01,  0.0000e+00,  1.0000e+00,  1.0000e+00 },
    {  4.0569e-01, -3.4859e+01,  2.7762e-03,  9.9758e-01 },
    {  2.1099e-01, -1.0207e-01,  1.6655e+00,  4.2542e+00 },
    {  2.0000e-01,  2.7118e-01,  6.1110e-01,  1.0454e+00 },
    {  5.5325e-01, -7.0746e-02,  3.5839e+00,  4.3084e+00 },
    {  7.3839e-01, -1.1350e-01,  6.3423e-01,  2.1142e+00 }
  };

  //lateral parameters
  const int Nrpar = 4;
  const G4double rpar[4*Nrpar][2] = {
    {  1.6065e+00, -1.9118e-01}, //Ecal for showerType = 1
    {  8.3070e+00, -1.2512e+00},
    {  6.2098e-01, -2.9482e-02},
    { -1.6002e-01,  1.9410e-02},
    {  2.1779e+01, -2.6719e+00}, //Hcal for showerType = 1
    { -1.1468e-01, -1.2217e-01},
    {  4.4093e-01, -1.5766e-02},
    { -2.3089e-02,  9.3034e-03},
    {  1.3713e+01, -1.9910e+00}, //Hcal for showerType = 2
    { -1.4097e+00,  7.2315e-02},
    {  4.4531e-01, -2.8746e-03},
    {  3.1108e-02, -4.1437e-04},
    {  1.9392e+00,  3.4218e-02}, //Hcal for showerType = 3
    {  7.8122e+00, -1.5978e+00},
    {  6.6561e-01, -3.9986e-02},
    { -2.5633e-01,  5.7796e-02}
  };

}

#endif
