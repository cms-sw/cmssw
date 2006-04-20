#ifndef SimG4Core_CustomPDGParser_H
#define SimG4Core_CustomPDGParser_H

#include <vector>

class CustomPDGParser
{
public:
    static bool s_isRHadron(int pdg);
    static bool s_isSLepton(int pdg);
    static bool s_isRBaryon(int pdg);
    static bool s_isRMeson(int pdg);
    static bool s_isRGlueball(int pdg);
    static double s_charge(int pdg);
    static double s_spin(int pdg);
    static std::vector<int> s_containedQuarks(int pdg);
    static int s_containedQuarksCode(int pdg);
};

#endif
