#ifndef SimG4Core_CustomPhysics_CustomPDGParser_h
#define SimG4Core_CustomPhysics_CustomPDGParser_h

#include <vector>

class CustomPDGParser {
public:
  static bool s_isgluinoHadron(int pdg);
  static bool s_isstopHadron(int pdg);
  static bool s_issbottomHadron(int pdg);
  static bool s_isSLepton(int pdg);
  static bool s_isRBaryon(int pdg);
  static bool s_isRMeson(int pdg);
  static bool s_isMesonino(int pdg);
  static bool s_isSbaryon(int pdg);
  static bool s_isRGlueball(int pdg);
  static bool s_isDphoton(int pdg);
  static bool s_isChargino(int pdg);
  static bool s_isSIMP(int pdg);
  static double s_charge(int pdg);
  static double s_spin(int pdg);
  static std::vector<int> s_containedQuarks(int pdg);
  static int s_containedQuarksCode(int pdg);
};

#endif
