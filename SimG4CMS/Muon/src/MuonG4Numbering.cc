#include "SimG4CMS/Muon/interface/MuonG4Numbering.h"
#include "CondFormats/GeometryObjects/interface/MuonOffsetMap.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DD4hep/Filter.h"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4Step.hh"

#include <iostream>

//#define EDM_ML_DEBUG

MuonG4Numbering::MuonG4Numbering(const MuonGeometryConstants& muonConstants, const MuonOffsetMap* offMap, bool dd4hep)
    : offMap_(offMap), dd4hep_(dd4hep) {
  theLevelPart = muonConstants.getValue("level");
  theSuperPart = muonConstants.getValue("super");
  theBasePart = muonConstants.getValue("base");
  theStartCopyNo = muonConstants.getValue("xml_starts_with_copyno");

  // some consistency checks

  if (theBasePart != 1) {
    edm::LogVerbatim("MuonSim") << "MuonGeometryNumbering finds unusual base constant:" << theBasePart;
  }
  if (theSuperPart < 100) {
    edm::LogVerbatim("MuonSim") << "MuonGeometryNumbering finds unusual super constant:" << theSuperPart;
  }
  if (theLevelPart < 10 * theSuperPart) {
    edm::LogVerbatim("MuonSim") << "MuonGeometryNumbering finds unusual level constant:" << theLevelPart;
  }
  if ((theStartCopyNo != 0) && (theStartCopyNo != 1)) {
    std::cout << "MuonGeometryNumbering finds unusual start value for copy numbers:" << theStartCopyNo << std::endl;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonSim") << "StartCopyNo = " << theStartCopyNo;
  edm::LogVerbatim("MuonSim") << "MuonG4Numbering configured with"
                              << "Level = " << theLevelPart << " Super = " << theSuperPart << " Base = " << theBasePart
                              << " StartCopyNo = " << theStartCopyNo;
  edm::LogVerbatim("MuonSim") << "dd4hep flag set to " << dd4hep_ << " and offsetmap at " << offMap_;
#endif
}

MuonBaseNumber MuonG4Numbering::PhysicalVolumeToBaseNumber(const G4Step* aStep) {
  MuonBaseNumber num;
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();

  for (int ii = 0; ii < touch->GetHistoryDepth(); ii++) {
    G4VPhysicalVolume* vol = touch->GetVolume(ii);
    int copyno = vol->GetCopyNo();
    int extra(0);
    if (dd4hep_ && (offMap_ != nullptr)) {
      std::string namx = static_cast<std::string>(dd4hep::dd::noNamespace(vol->GetName()));
      std::size_t last = namx.rfind('_');
      std::string name = ((last == std::string::npos) ? namx : (namx.substr(0, last)));
      auto itr = offMap_->muonMap_.find(name);
      if (itr != offMap_->muonMap_.end())
        extra = (itr->second).first + (itr->second).second;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonSim") << "MuonG4Numbering: " << namx << ":" << name << " iterator "
                                  << (itr != offMap_->muonMap_.end()) << " Extra " << extra;
#endif
    }
    copyno += extra;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonSim") << "MuonG4Numbering: " << vol->GetName() << " " << copyno << " Split "
                                << copyNoRelevant(copyno) << ":" << theLevelPart << ":" << theSuperPart << " ";
#endif
    if (copyNoRelevant(copyno)) {
      num.addBase(getCopyNoLevel(copyno), getCopyNoSuperNo(copyno), getCopyNoBaseNo(copyno) - theStartCopyNo);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonSim") << " NoLevel " << getCopyNoLevel(copyno) << " Super " << getCopyNoSuperNo(copyno)
                                  << " Base " << getCopyNoBaseNo(copyno) << " Start " << theStartCopyNo;
#endif
    }
  }

  return num;
}

const int MuonG4Numbering::getCopyNoLevel(const int copyno) { return copyno / theLevelPart; }

const int MuonG4Numbering::getCopyNoSuperNo(const int copyno) { return (copyno % theLevelPart) / theSuperPart; }

const int MuonG4Numbering::getCopyNoBaseNo(const int copyno) { return copyno % theSuperPart; }

const bool MuonG4Numbering::copyNoRelevant(const int copyno) { return (copyno / theLevelPart) > 0; }
