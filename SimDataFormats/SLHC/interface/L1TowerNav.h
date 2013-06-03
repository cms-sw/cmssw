#ifndef SimDataFormats_SLHC_L1TowerNav
#define SimDataFormats_SLHC_L1TowerNav

//proto type, probably should be a namespace but I might extend it
//quick dirty hack to get the towers navigating correctly

class L1TowerNav {
public:
  // static const int kNullIEta=0; //return value if offset brings it to an invalid eta position (now removed, we return invalid iEta positions
  static const int kNullIPhi=0; //return value if we have an invalid phi position, note we can only achieve this if the starting iPhi is invalid

  static const int kIPhiMax=72;
  static const int kIEtaAbsHEMax=28; //end of HE, useful as the phi scale changes for HF
  static const int kIEtaAbsHFMax=32; //pTDR says 41 but we appear to end at 32, are we combining the high eta towers? check this! 
  static const int kHFIPhiScale=4; //each phi tower in HF is 4 that of HBHE

  static int getOffsetIEta(int iEta,int offset); //returns the iEta which is offset towers away form iEta
  static int getOffsetIPhi(int iEta,int iPhi,int offset); //returns the iPhi which is offset towers away from iPhi, needs iEta to know if in HBHE or HF
  static int getOffsetIPhiHBHE(int iPhi,int offset);
  static int getOffsetIPhiHF(int iPhi,int offset);
};


#endif
