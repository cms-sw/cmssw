// -*- C++ -*-
//
// Package:    TauNtuple
// Class:      TauDecay
// 
/**\class TauDecay TauDecay.cc TauDataFormat/TauNtuple/src/TauDecay.cc

 Description: This class reconstructs the MODE modes of the Tauola decays and provides a bit mask of the decay structure for the tau

*/
//
// Original Author:  Ian Nugent  
//         Created:  Fri Nov 18 13:49:02 CET 2011
//
//
#ifndef TauDecay_h
#define TauDecay_h

#include <string>

class TauDecay {
 public:
  // TAUOLA list of decay modes avalible presently available in Tauola are (MODE):  
  //                                                              
  // * DEC    BRTAU    NORMAL    ROUTINE    CHANNEL         * 
  // *   1  0.17810  0.17810     DADMEL     ELECTRON        * 
  // *   2  0.17370  0.17370     DADMMU     MUON            * 
  // *   3  0.11080  0.11080     DADMPI     PION            * 
  // *   4  0.25320  0.25320     DADMRO     RHO (->2PI)     *
  // *   5  0.18250  0.18250     DADMAA     A1  (->3PI)     * 
  // *   6  0.00710  0.00710     DADMKK     KAON            *
  // *   7  0.01280  0.01280     DADMKS     K*              * 
  // *   8  0.04500  0.04500     DAD4PI     2PI- PI+ PI0    *
  // *   9  0.01230  0.01230     DAD4PI     PI- 3PI0        *
  // *  10  0.00500  0.00500     DADNPI     2PI- PI+ 2PI0   *
  // *  11  0.00080  0.00080     DADNPI     3PI- 2PI+       *
  // *  12  0.00020  0.00020     DADNPI     3PI- 2PI+ PI0   * 
  // *  13  0.00030  0.00030     DADNPI     2PI- PI+ 3PI0   *
  // *  14  0.00190  0.00190     DADMPK     K+ K- PI+       *  
  // *  15  0.00120  0.00120     DADMPK     K0B K0 PI+      * 
  // *  16  0.00300  0.00300     DADMPK     K+ K0B PI0      *  
  // *  17  0.00100  0.00100     DADMPK     K+ PI0 PI0      * 
  // *  18  0.00230  0.00230     DADMPK     K+ PI- PI+      *
  // *  19  0.00390  0.00390     DADMPK     K0 PI0 PI+      * 
  // *  20  0.00170  0.00170     DADMPK     ET PI- PI0      * 
  // *  21  0.00160  0.00160     DADMPK     PI-PI0 GAM      * (obsolete ... merged with 4 do to pythia and photos radiation)
  // *  22  0.00160  0.00160     DADMPK     K- K0B GAM      * (ignore photon?)
  /////////////////////////////////////////////////////////////////////////////////////////////
  // List of Pythia8 Modes
  //    3        0     0   0.1076825 1521       16     -211
  //    6        1     0   0.0069601 1521       16     -321
  //    1        2     0   0.1772832 1531       16       11      -12
  //    2        3     0   0.1731072 1531       16       13      -14
  //    4        4     0   0.2537447 1532       16      111     -211
  //    22       5     0   0.0015809 1532       16      311     -321
  //    Keta     6     0   0.0001511 1532       16      221     -321
  //    7        7     0   0.0083521 1533       16     -211     -311
  //    7        8     0   0.0042655 1533       16      111     -321
  //    5      0   0.0924697 1541       16      111      111     -211
  //    5    10     1   0.0925691 1543       16     -211     -211      211
  //    19      11     0   0.0039772 1542       16      111     -211     -311
  //    18      12     0   0.0034701 1542       16     -211      211     -321
  //    14      13     0   0.0014318 1542       16     -211     -321      321
  //    16      14     0   0.0015809 1542       16      111      311     -321
  //    15      15     0   0.0011932 1542       16      130     -211      310
  //    17      16     0   0.0006463 1542       16      111      111     -321
  //    15      17     0   0.0002386 1542       16      130      130     -211
  //    15      18     0   0.0002386 1542       16     -211      310      310
  //    20      19     0   0.0013821 1543       16      111     -211      221
  //    21->4   20     0   0.0017520 1544       16       22      111     -211 (obsolete ... merged with 4 do to pythia and photos radiation)
  //    8       21     0   0.0459365 1551       16      111     -211     -211      211
  //    9       22     0   0.0104401 1551       16      111      111      111     -211
  //    10      23     0   0.0049069 1561       16      111      111     -211     -211      211
  //    25      24     0   0.0009515 1561       16      111      111      111      111     -211
  //    11      25     0   0.0008342 1561       16     -211     -211     -211      211      211
  //    26      26     0   0.0001631    0       16     -211     -211      211      221
  //    27      27     0   0.0001491    0       16      111      111     -211      221
  //    28      28     0   0.0001392    0       16      111      111     -211      223
  //    29      29     0   0.0001193    0       16     -211     -211      211      223
  //    30      30     0   0.0004077    0       16      223     -321
  //    31      31     0   0.0004773    0       16      111      111      111     -321
  //    32      32     0   0.0003052    0       16      111     -211      211     -321
  //    33      33     0   0.0002784    0       16      221     -323
  //    34      34     0   0.0002366    0       16      111      111     -211     -311
  //    35      35     0   0.0002237    0       16     -211     -211      211     -311
  //    36      36     0   0.0002953    0       16      111     -211     -311      311
  //    37      37     0   0.0000590    0       16      111     -211     -321      321
        
  enum MODE{MODE_UNKNOWN=0,
	    MODE_ELECTRON,
	    MODE_MUON,
	    MODE_PION,
	    MODE_PIPI0,
	    MODE_3PI,
            MODE_PI2PI0,
	    MODE_KAON,
	    MODE_K0PI,
	    MODE_KPI0,
	    MODE_3PIPI0,
	    MODE_PI3PI0,
	    MODE_3PI2PI0,
	    MODE_5PI,
	    MODE_5PIPI0,
	    MODE_3PI3PI0,
	    MODE_KPIK,
	    MODE_K0BK0PI,
	    MODE_KK0BPI0,
	    MODE_K2PI0,
	    MODE_KPIPI,
	    MODE_PIK0PI0,
	    MODE_ETAPIPI0,
	    MODE_PIPI0GAM,
	    MODE_KK0B,
	    MODE_PI4PI0,
	    MODE_3PIETA,
	    MODE_PI2PI0ETA,
	    MODE_PI2PI0OMEGA,
	    MODE_3PIOMEGA,
	    MODE_KOMEGA,
	    MODE_K3PI0,
	    MODE_K2PIPI0,
	    MODE_KETA,
	    MODE_K0PI2PI0,
	    MODE_K03PI,
	    MODE_2K0PIPI0,
	    MODE_KPIKPI0,
	    NMODEID
  };
  
  
  enum TauDecayStructure{other=0,
			 OneProng=1,
			 ThreeProng=2,
			 FiveProng=4,
			 OnePi0=8,
			 TwoPi0=32,
			 ThreePi0=64,
			 Res_a1_pm=128,
			 Res_a1_0=256,
			 Res_rho_pm=512,
			 Res_rho_0=1024,
			 Res_eta=2048,
			 Res_omega=4096,
			 Res_Kstar_pm=8192,
			 Res_Kstar_0=16384,
			 KS0_to_pipi=32768
  };
  
  TauDecay();
  ~TauDecay();
  
  void Reset();
  bool isTauFinalStateParticle(int pdgid);
  bool isTauParticleCounter(int pdgid);
  bool isTauResonanceCounter(int pdgid);
  void ClassifyDecayMode(unsigned int &MODE_ID,unsigned int &TauBitMask);
  unsigned int nProng(unsigned int &TauBitMask);
  unsigned int nPi0(unsigned int &TauBitMask);
  bool hasResonance(unsigned int &TauBitMask, int pdgid);
  static std::string DecayMode(unsigned int &MODE_ID);
 
 private:
  // Functions
  void ClassifyDecayResonance(unsigned int &TauBitMask);
  
  //counting varibles
  unsigned int n_pi,n_pi0,n_K,n_K0L,n_K0S,n_gamma,n_nu,n_e,n_mu; // particle counters
  unsigned int n_a1,n_a10,n_rho,n_rho0,n_eta,n_omega,n_Kstar0,n_Kstar,unknown; // resonance counters
  
};
#endif
