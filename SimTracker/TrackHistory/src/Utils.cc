#include "SimTracker/TrackHistory/interface/Utils.h"

unsigned short mapG4toCMSProcType(unsigned int i)
    {
                                        // Geant4Process -> CmsProcess
        if (i == 0)
        {
            return 2;                   // undefined geant process -> "Primary"
        }
        else if (i >= 1 && i < 25)
        {
            switch (i)
            {
                case 1: return 1;       // "CoulombScat" -> "Unknown" => DOUBLE-CHECK THIS
                case 2: return 7;       // "Ionisation" -> "EIoni" => what about "MuIoni"?
                case 3: return 13;      // "Brems" -> "EBrem" => what about "MuBrem"?
                case 4: return 11;      // "PairProdCharged" -> "MuPairProd" => DOUBLE-CHECK THIS
                case 5: return 6;       // "Annih" -> "Annihilation"
                case 6: return 6;       // "AnnihToMuMu" -> "Annihilation"
                case 7: return 6;       // "AnnihToHad" -> "Annihilation"
                case 8: return 3;       // "NuclearStopp" -> "Hadronic" => DOUBLE-CHECK THIS
                case 10: return 1;      // "Msc" -> "Unknown" => DOUBLE-CHECK THIS
                case 11: return 1;      // "Rayleigh" -> "Unknown" => DOUBLE-CHECK THIS
                case 12: return 10;     // "PhotoElectric" -> "Photon"
                case 13: return 5;      // "Compton" -> "Compton"
                case 14: return 12;     // "Conv" -> "Conversions"
                case 15: return 12;     // "ConvToMuMu" -> "Conversions"
                case 21: return 1;      // "Cerenkov" -> "Unknown" => DOUBLE-CHECK THIS
                case 22: return 1;      // "Scintillation" -> "Unknown" => DOUBLE-CHECK THIS
                case 23: return 14;     // "SynchRad" -> "SynchrotronRadiation"
                case 24: return 14;     // "TransRad" -> "SynchrotronRadiation" => DOUBLE-CHECK THIS
            }
        }
        else if (i == 91 || i == 92)
        {
            return 1;                   // "Transportation", "CoupleTrans" -> "Unknown" => DOUBLE-CHECK THIS
        }
        else if (i >= 31 && i < 58)
        {
            return 1;                   // "OpAbsorp",  "OpBoundary",  "OpRayleigh",  "OpWLS",  "OpMieHG",
                                        // "DNAElastic",  "DNAExcit",  "DNAIonisation",  "DNAVibExcit",  "DNAAttachment",
                                        // "DNAChargeDec",  "DNAChargeInc" -> "Unknown"
                                        //  => DOUBLE-CHECK THIS
        }
        else if (i >= 111 && i < 162)
        {
            return 3;                   // "HadElastic",  "HadInElastic",  "HadCaptue", "HadFission",  "HadAtRest",  "HadCEX"
                                        // -> "Hadronic"
        }
        else if (i >= 201 && i < 232)
        {
            return 4;                   // "Decay",  "DecayWSpin", "DecayPiWSpin",  "DecayRadio",  "DecayUnKnown",  "DecayExt"
                                        // -> "Decay" => Geant4 decays
        }
        else if (i == 401 || i == 402)
        {
            return 1;                   // "StepLimiter", "UsrSpecCuts" -> "Unknown" => DOUBLE-CHECK THIS
        }

        return 0;                   // otherwise return "UndefinedProcess"}
    }
