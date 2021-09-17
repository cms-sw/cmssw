#include <SimG4Core/CustomPhysics/interface/CustomPDGParser.h>
#include <cstdlib>

// check for R-hadron with gluino content
bool CustomPDGParser::s_isgluinoHadron(int pdg) {
  int pdgAbs = abs(pdg);
  return ((pdgAbs % 100000 / 10000 == 9) || (pdgAbs % 10000 / 1000 == 9) || s_isRGlueball(pdg));
}

bool CustomPDGParser::s_isstopHadron(int pdg) {
  int pdgAbs = abs(pdg);
  return ((pdgAbs % 10000 / 1000 == 6) || (pdgAbs % 1000 / 100 == 6));
}

bool CustomPDGParser::s_issbottomHadron(int pdg) {
  int pdgAbs = abs(pdg);
  return ((pdgAbs % 10000 / 1000 == 5) || (pdgAbs % 10000 / 100 == 5));
}

bool CustomPDGParser::s_isSLepton(int pdg) {
  int pdgAbs = abs(pdg);
  return (pdgAbs / 100 % 10000 == 0 && pdgAbs / 10 % 10 == 1);
}

bool CustomPDGParser::s_isRBaryon(int pdg) {
  int pdgAbs = abs(pdg);
  return (pdgAbs % 100000 / 10000 == 9);
}

bool CustomPDGParser::s_isRGlueball(int pdg) {
  int pdgAbs = abs(pdg);
  return (pdgAbs % 100000 / 10 == 99);
}

bool CustomPDGParser::s_isDphoton(int pdg) {
  int pdgAbs = abs(pdg);
  return (pdgAbs == 1072000) || (pdgAbs == 1023);
}

bool CustomPDGParser::s_isRMeson(int pdg) {
  int pdgAbs = abs(pdg);
  return (pdgAbs % 10000 / 1000 == 9);
}

bool CustomPDGParser::s_isMesonino(int pdg) {
  int pdgAbs = abs(pdg);
  return ((pdgAbs % 10000 / 100 == 6) || (pdgAbs % 10000 / 100 == 5));
}

bool CustomPDGParser::s_isSbaryon(int pdg) {
  int pdgAbs = abs(pdg);
  return ((pdgAbs % 10000 / 1000 == 6) || (pdgAbs % 10000 / 1000 == 5));
}

bool CustomPDGParser::s_isChargino(int pdg) {
  int pdgAbs = abs(pdg);
  return (pdgAbs == 1000024);
}

bool CustomPDGParser::s_isSIMP(int pdg) {
  int pdgAbs = abs(pdg);
  return (pdgAbs == 9000006);
}

double CustomPDGParser::s_charge(int pdg) {
  float charge = 0, sign = 1;
  int pdgAbs = abs(pdg);
  if (pdg < 0)
    sign = -1;

  if (s_isSLepton(pdg))  //Sleptons
  {
    if (pdgAbs % 2 == 0)
      return 0;
    else
      return -sign;
  }

  if (s_isDphoton(pdg)) {
    return charge;
  }
  if (s_isChargino(pdg)) {
    return sign;
  }
  if (s_isSIMP(pdg)) {
    return 0;
  }
  if (s_isRMeson(pdg)) {
    std::vector<int> quarks = s_containedQuarks(pdg);
    if ((quarks[1] % 2 == 0 && quarks[0] % 2 == 1) || (quarks[1] % 2 == 1 && quarks[0] % 2 == 0))
      charge = 1;
    charge *= sign;
    return charge;
  }

  if (s_isRBaryon(pdg)) {
    int baryon = s_containedQuarksCode(pdg);
    for (int q = 1; q < 1000; q *= 10) {
      if (baryon / q % 2 == 0)
        charge += 2;
      else
        charge -= 1;
    }
    charge /= 3;
    charge *= sign;
    return charge;
  }

  if (s_isMesonino(pdg)) {
    int quark = s_containedQuarks(pdg)[0];
    int squark = abs(pdg / 100 % 10);
    if (squark % 2 == 0 && quark % 2 == 1)
      charge = 1;
    if (squark % 2 == 1 && quark % 2 == 0)
      charge = 1;
    charge *= sign;
    if (s_issbottomHadron(pdg))
      charge *= -1;
    return charge;
  }

  if (s_isSbaryon(pdg)) {
    int baryon = s_containedQuarksCode(pdg) + 100 * (abs(pdg / 1000 % 10));  //Adding the squark back on
    for (int q = 1; q < 1000; q *= 10) {
      if (baryon / q % 2 == 0)
        charge += 2;
      else
        charge -= 1;
    }
    charge /= 3;
    charge *= sign;
    if (s_issbottomHadron(pdg))
      charge *= -1;
    return charge;
  }

  return 0;
}

double CustomPDGParser::s_spin(int pdg) {
  // The PDG numbering is described in the Review of Particle Physics:
  // "3. In composite quark systems (diquarks, mesons, and baryons) ...
  // the rightmost digit nJ = 2J + 1 gives the system's spin."
  // Since this does not apply to SUSY / exotic particles,
  // if the spin is important for the simulation
  // it should be hard-coded based on PDG ID in this function.
  int pdgAbs = abs(pdg);
  return pdgAbs % 10;
}

std::vector<int> CustomPDGParser::s_containedQuarks(int pdg) {
  std::vector<int> quarks;
  for (int i = s_containedQuarksCode(pdg); i > 0; i /= 10) {
    quarks.push_back(i % 10);
  }
  return quarks;
}

int CustomPDGParser::s_containedQuarksCode(int pdg) {
  int pdgAbs = abs(pdg);
  if (s_isRBaryon(pdg))
    return pdgAbs / 10 % 1000;

  if (s_isRMeson(pdg))
    return pdgAbs / 10 % 100;

  if (s_isMesonino(pdg))
    return pdgAbs / 10 % 1000 % 10;

  if (s_isSbaryon(pdg))
    return pdgAbs / 10 % 1000 % 100;

  return 0;
}
