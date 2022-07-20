#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

static const int nprocesses = 49;
static const std::string g4processes[nprocesses] = {
    "Primary",         "Transportation", "CoupleTrans",  "CoulombScat",    "Ionisation",    "Brems",
    "PairProdCharged", "Annih",          "AnnihToMuMu",  "AnnihToHad",     "NuclearStopp",  "Msc",
    "Rayleigh",        "PhotoElectric",  "Compton",      "Conv",           "ConvToMuMu",    "Cerenkov",
    "Scintillation",   "SynchRad",       "TransRad",     "OpAbsorp",       "OpBoundary",    "OpRayleigh",
    "OpWLS",           "OpMieHG",        "MuDBrem",      "MuMuonPairProd", "DNAIonisation", "DNAVibExcit",
    "DNAAttachment",   "DNAChargeDec",   "DNAChargeInc", "HadElastic",     "HadInelastic",  "HadCapture",
    "HadFission",      "HadAtRest",      "HadCEX",       "Decay",          "DecayWSpin",    "DecayPiWSpin",
    "DecayRadio",      "DecayUnKnown",   "DecayExt",     "GFlash",         "StepLimiter",   "UsrSpecCuts",
    "NeutronKiller"};
static const int g4subtype[nprocesses] = {
    0,   // Primary generator
    91,  // Transportation
    92,  // CoupleTrans
    1,   // CoulombScat
    2,   // Ionisation
    3,   // Brems
    4,   // PairProdCharged
    5,   // Annih
    6,   // AnnihToMuMu
    7,   // AnnihToHad
    8,   // NuclearStopp
    10,  // Msc
    11,  // Rayleigh
    12,  // PhotoElectric
    13,  // Compton
    14,  // Conv
    15,  // ConvToMuMu
    21,  22, 23, 24, 31, 32, 33, 34, 35,
    40,  // muDBrem
    49,  // MuMuonPairProd
    53,  54, 55, 56, 57,
    111,  // HadElastic
    121,  // HadInelastic
    131,  // HadCapture
    141,  // HadFission
    151,  // HadAtRest
    161,
    201,  // Decay
    202,  // DecayWSpin
    203,  // DecayPiWSpin
    210,  // DecayRadio
    211,  // DecayUnKnown
    231,  // DecayExt
    301,  // GFlash
    401,  // StepLimiter
    402,
    403  // NeutronKiller
};

G4ProcessTypeEnumerator::G4ProcessTypeEnumerator() {}

G4ProcessTypeEnumerator::~G4ProcessTypeEnumerator() {}

std::string G4ProcessTypeEnumerator::processG4Name(int idx) const {
  std::string res = "";
  for (int i = 0; i < nprocesses; ++i) {
    if (idx == g4subtype[i]) {
      res = g4processes[i];
      break;
    }
  }
  return res;
}

int G4ProcessTypeEnumerator::processId(const std::string& name) const {
  int idx = 0;
  for (int i = 0; i < nprocesses; ++i) {
    if (name == g4processes[i]) {
      idx = g4subtype[i];
      break;
    }
  }
  return idx;
}
