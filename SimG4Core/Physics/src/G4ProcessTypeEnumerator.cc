#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

static const int nprocesses = 46;
static const std::string g4processes[nprocesses] = { 
  "Transportation",  "CoupleTrans",  "CoulombScat",  "Ionisation",  "Brems",
  "PairProdCharged",  "Annih",  "AnnihToMuMu",  "AnnihToHad",  "NuclearStopp",
  "Msc",  "Rayleigh",  "PhotoElectric",  "Compton",  "Conv",
  "ConvToMuMu",  "Cerenkov",  "Scintillation",  "SynchRad",  "TransRad",
  "OpAbsorp",  "OpBoundary",  "OpRayleigh",  "OpWLS",  "OpMieHG",
  "DNAElastic",  "DNAExcit",  "DNAIonisation",  "DNAVibExcit",  "DNAAttachment",
  "DNAChargeDec",  "DNAChargeInc",  "HadElastic",  "HadInElastic",  "HadCaptue",
  "HadFission",  "HadAtRest",  "HadCEX",  "Decay",  "DecayWSpin",
  "DecayPiWSpin",  "DecayRadio",  "DecayUnKnown",  "DecayExt",  "StepLimiter",
  "UsrSpecCuts" }; 
static const int g4subtype[nprocesses] = { 
  91,  92,  1,  2,  3,  4,  5,  6,  7,  8,
  10,  11,  12,  13,  14,  15,  21,  22,  23,  24,
  31,  32,  33,  34,  35,  51,  52,  53,  54,  55,
  56,  57,  111,  121,  131,  141,  151,  161,  201,  202,
  203,  210,  211,  231,  401,  402 }; 


G4ProcessTypeEnumerator::G4ProcessTypeEnumerator()
{}

G4ProcessTypeEnumerator::~G4ProcessTypeEnumerator()
{}

std::string G4ProcessTypeEnumerator::processG4Name(int idx)
{
  std::string res = "";
  for(int i=0; i<nprocesses; ++i) {
    if(idx == g4subtype[i]) {
      res = g4processes[i];
      break;
    }
  }
  return res;
}

