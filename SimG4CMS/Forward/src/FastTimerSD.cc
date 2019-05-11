#include "SimG4CMS/Forward/interface/FastTimerSD.h"

#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"

#include <vector>

#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"

#include <iostream>

//#define EDM_ML_DEBUG
//-------------------------------------------------------------------
FastTimerSD::FastTimerSD(const std::string& name,
                         const DDCompactView& cpv,
                         const SensitiveDetectorCatalog& clg,
                         edm::ParameterSet const& p,
                         const SimTrackManager* manager)
    : TimingSD(name, cpv, clg, p, manager), ftcons(nullptr) {
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("FastTimerSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");

  SetVerboseLevel(verbn);

  std::string attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, name, 0)};
  DDFilteredView fv(cpv, filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());
  std::vector<int> temp = dbl_to_int(getDDDArray("Type", sv));
  type_ = temp[0];

  setTimeFactor(100.);

  edm::LogInfo("FastTimerSim") << "FastTimerSD: Instantiation completed for " << name << " of type " << type_;
}

FastTimerSD::~FastTimerSD() {}

uint32_t FastTimerSD::setDetUnitId(const G4Step* aStep) {
  //Find the depth segment
  const G4ThreeVector& global = getGlobalEntryPoint();
  const G4ThreeVector& local = getLocalEntryPoint();
  int iz = (global.z() > 0) ? 1 : -1;
  std::pair<int, int> izphi = ((ftcons) ? ((type_ == 1) ? (ftcons->getZPhi(std::abs(local.z()), local.phi()))
                                                        : (ftcons->getEtaPhi(local.perp(), local.phi())))
                                        : (std::pair<int, int>(0, 0)));
  uint32_t id = FastTimeDetId(type_, izphi.first, izphi.second, iz).rawId();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FastTimerSD") << "Volume " << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetName() << ": "
                                  << global.z() << " Iz(eta)phi " << izphi.first << ":" << izphi.second << ":" << iz
                                  << " id " << std::hex << id << std::dec;
#endif
  return id;
}

void FastTimerSD::update(const BeginOfJob* job) {
  const edm::EventSetup* es = (*job)();
  edm::ESHandle<FastTimeDDDConstants> fdc;
  es->get<IdealGeometryRecord>().get(fdc);
  if (fdc.isValid()) {
    ftcons = &(*fdc);
  } else {
    edm::LogError("FastTimerSim") << "FastTimerSD : Cannot find FastTimeDDDConstants";
    throw cms::Exception("Unknown", "FastTimerSD") << "Cannot find FastTimeDDDConstants\n";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("FastTimerSD") << "FastTimerSD::Initialized with FastTimeDDDConstants\n";
#endif
}

std::vector<double> FastTimerSD::getDDDArray(const std::string& str, const DDsvalues_type& sv) {
  DDValue value(str);
  if (DDfetch(&sv, value)) {
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("FastTimerSim") << "FastTimerSD : # of " << str << " bins " << nval << " < 1 ==> illegal";
      throw cms::Exception("DDException") << "FastTimerSD: cannot get array " << str;
    }
    return fvec;
  } else {
    edm::LogError("FastTimerSim") << "FastTimerSD: cannot get array " << str;
    throw cms::Exception("DDException") << "FastTimerSD: cannot get array " << str;
  }
}
