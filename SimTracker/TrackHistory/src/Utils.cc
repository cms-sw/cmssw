#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimTracker/TrackHistory/interface/CMSProcessTypes.h"
#include "SimTracker/TrackHistory/interface/Utils.h"

G4toCMSLegacyProcTypeMap::G4toCMSLegacyProcTypeMap()
{
  // Geant4Process -> CmsProcess
  m_map[0]   = CMS::Primary;              // "Primary"         -> "Primary"
  m_map[91]  = CMS::Unknown;              // "Transportation"  -> "Unknown"
  m_map[92]  = CMS::Unknown;              // "CoupleTrans"     -> "Unknown" => DOUBLE-CHECK THIS
  m_map[1]   = CMS::Unknown;              // "CoulombScat"     -> "Unknown" => DOUBLE-CHECK THIS
  m_map[2]   = CMS::EIoni;                // "Ionisation"      -> "EIoni" => WHAT ABOUT MuIoni?
  m_map[3]   = CMS::EBrem;                // "Brems"           -> "EBrem" => WHAT ABOUT MuBrem?
  m_map[4]   = CMS::MuPairProd;           // "PairProdCharged" -> "MuPairProd" => DOUBLE-CHECK THIS
  m_map[5]   = CMS::Annihilation;         // "Annih"           -> "Annihilation"
  m_map[6]   = CMS::Annihilation;         // "AnnihToMuMu"     -> "Annihilation"
  m_map[7]   = CMS::Annihilation;         // "AnnihToHad"      -> "Annihilation"
  m_map[8]   = CMS::Hadronic;             // "NuclearStopp"    -> "Hadronic" => DOUBLE-CHECK THIS
  m_map[10]  = CMS::Unknown;              // "Msc"             -> "Unknown" => DOUBLE-CHECK THIS
  m_map[11]  = CMS::Unknown;              // "Rayleigh"        -> "Unknown" => DOUBLE-CHECK THIS
  m_map[12]  = CMS::Photon;               // "PhotoElectric"   -> "Photon"
  m_map[13]  = CMS::Compton;              // "Compton"         -> "Compton"
  m_map[14]  = CMS::Conversions;          // "Conv"            -> "Conversions"
  m_map[15]  = CMS::Conversions;          // "ConvToMuMu"      -> "Conversions"
  m_map[21]  = CMS::Unknown;              // "Cerenkov"        -> "Unknown" => DOUBLE-CHECK THIS
  m_map[22]  = CMS::Unknown;              // "Scintillation"   -> "Unknown" => DOUBLE-CHECK THIS
  m_map[23]  = CMS::SynchrotronRadiation; // "SynchRad"        -> "SynchrotronRadiation"
  m_map[24]  = CMS::SynchrotronRadiation; // "TransRad"        -> "SynchrotronRadiation" => DOUBLE-CHECK THIS
  m_map[31]  = CMS::Unknown;              // "OpAbsorp"        -> "Unknown" => NOT USED IN CMS
  m_map[32]  = CMS::Unknown;              // "OpBoundary"      -> "Unknown" => NOT USED IN CMS
  m_map[33]  = CMS::Unknown;              // "OpRayleigh"      -> "Unknown" => NOT USED IN CMS
  m_map[34]  = CMS::Unknown;              // "OpWLS"           -> "Unknown" => NOT USED IN CMS
  m_map[35]  = CMS::Unknown;              // "OpMieHG"         -> "Unknown" => NOT USED IN CMS
  m_map[51]  = CMS::Unknown;              // "DNAElastic"      -> "Unknown" => NOT USED IN CMS
  m_map[52]  = CMS::Unknown;              // "DNAExcit"        -> "Unknown" => NOT USED IN CMS
  m_map[53]  = CMS::Unknown;              // "DNAIonisation"   -> "Unknown" => NOT USED IN CMS
  m_map[54]  = CMS::Unknown;              // "DNAVibExcit"     -> "Unknown" => NOT USED IN CMS
  m_map[55]  = CMS::Unknown;              // "DNAAttachment"   -> "Unknown" => NOT USED IN CMS
  m_map[56]  = CMS::Unknown;              // "DNAChargeDec"    -> "Unknown" => NOT USED IN CMS
  m_map[57]  = CMS::Unknown;              // "DNAChargeInc"    -> "Unknown" => NOT USED IN CMS
  m_map[111] = CMS::Hadronic;             // "HadElastic"      -> "Hadronic"
  m_map[121] = CMS::Hadronic;             // "HadInelastic"    -> "Hadronic"
  m_map[131] = CMS::Hadronic;             // "HadCapture"      -> "Hadronic"
  m_map[141] = CMS::Hadronic;             // "HadFission"      -> "Hadronic"
  m_map[151] = CMS::Hadronic;             // "HadAtRest"       -> "Hadronic"
  m_map[161] = CMS::Hadronic;             // "HadCEX"          -> "Hadronic"
  m_map[201] = CMS::Decay;                // "Decay"           -> "Decay"
  m_map[202] = CMS::Decay;                // "DecayWSpin"      -> "Decay"
  m_map[203] = CMS::Decay;                // "DecayPiWSpin"    -> "Decay"
  m_map[210] = CMS::Decay;                // "DecayRadio"      -> "Decay"
  m_map[211] = CMS::Decay;                // "DecayUnKnown"    -> "Decay"
  m_map[231] = CMS::Decay;                // "DecayExt"        -> "Decay"
  m_map[301] = CMS::Unknown;              // "GFlash"          -> "Unknown"
  m_map[401] = CMS::Unknown;              // "StepLimiter"     -> "Unknown"
  m_map[402] = CMS::Unknown;              // "UsrSpecCuts"     -> "Unknown"
  m_map[403] = CMS::Unknown;              // "NeutronKiller"   -> "Unknown"
}

const unsigned int G4toCMSLegacyProcTypeMap::processId(unsigned int g4ProcessId) const
{
  MapType::const_iterator it = m_map.find(g4ProcessId);

  if( it == m_map.end() )
  {
    edm::LogError("UnknownProcessType") << "Encountered an unknown process type: " << g4ProcessId << ". Mapping it to 'Undefined'.";
    return CMS::Undefined;
  }
  else
    return it->second;
}
