// -*- C++ -*-
//
// Package:    TauNtuple
// Class:      TauDecay
// 
/**\class TauDecay TauDecay.cc TauDataFormat/TauNtuple/src/TauDecay.cc

 Description: This class reconstructs the JAK modes of the Tauola decays and provides a bit mask of the decay structure for the tau

*/
//
// Original Author:  Ian Nugent  
//         Created:  Fri Nov 18 13:49:02 CET 2011
// $Id: TauDecay.h,v 1.1 2012/02/10 10:08:13 inugent Exp $
//
//
#ifndef TauDecay_h
#define TauDecay_h

//
// class declaration
//
class TauDecay {
 public:
  // TAUOLA list of decay modes avalible presently available in Tauola are (JAK):  
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
  // *  21  0.00160  0.00160     DADMPK     PI-PI0 GAM      *
  // *  22  0.00160  0.00160     DADMPK     K- K0B GAM      *
        
  enum JAK{JAK_UNKNOWN=0,
	   JAK_ELECTRON=1,
	   JAK_MUON=2,
	   JAK_PION=3,
	   JAK_RHO_PIPI0=4,
	   JAK_A1_3PI=5,
	   JAK_KAON=6,
	   JAK_KSTAR=7,
	   JAK_3PIPI0=8,
	   JAK_PI3PI0=9,
	   JAK_3PI2PI0=10,
	   JAK_5PI=11,
	   JAK_5PIPI0=12,
	   JAK_3PI3PI0=13,
	   JAK_KPIK=14,
	   JAK_K0BK0PI=15,
	   JAK_KK0BPI0=16,
	   JAK_K2PI0=17,
	   JAK_KPIPI=18,
	   JAK_PIK0PI0=19,
	   JAK_ETAPIPI0=20,
	   JAK_PIPI0GAM=21,
	   JAK_KK0B=22,
	   NJAKID=23
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
  void ClassifyDecayMode(unsigned int &JAK_ID,unsigned int &TauBitMask);
  unsigned int nProng(unsigned int &TauBitMask){
    if(OneProng&TauBitMask)   return 1;
    if(ThreeProng&TauBitMask) return 3;
    if(FiveProng&TauBitMask)  return 5;
    return 7;
  }
  unsigned int nPi0(unsigned int &TauBitMask){
    if(OnePi0&TauBitMask)   return 1;
    if(TwoPi0&TauBitMask)   return 2;
    if(ThreePi0&TauBitMask) return 3;
    return 0;
  }

  
 private:
  // Functions
  void ClassifyDecayResonance(unsigned int &TauBitMask);
  
  //counting varibles
  unsigned int n_pi,n_pi0,n_K,n_K0L,n_K0S,n_gamma,n_nu,n_e,n_mu; // particle counters
  unsigned int n_a1,n_a10,n_rho,n_rho0,n_eta,n_omega,n_Kstar0,n_Kstar,unknown; // resonance counters
  
};
#endif
