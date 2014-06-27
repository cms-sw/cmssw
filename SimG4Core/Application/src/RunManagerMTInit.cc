#include "SimG4Core/Application/interface/RunManagerMTInit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

RunManagerMTInit::RunManagerMTInit(const edm::ParameterSet& iConfig):
  m_p(iConfig),
  firstRun(true),
  m_pUseMagneticField(iConfig.getParameter<bool>("UseMagneticField"))
{}

RunManagerMTInit::~RunManagerMTInit() {}

RunManagerMTInit::ESProducts RunManagerMTInit::readES(const edm::EventSetup& iSetup) const {
  taskQueue_.pushAndWait([this, &iSetup] {
    bool geomChanged = idealGeomRcdWatcher_.check(iSetup);
    if (geomChanged && (!firstRun)) {
      throw cms::Exception("BadConfig") 
        << "[SimG4Core RunManager]\n"
        << "The Geometry configuration is changed during the job execution\n"
        << "this is not allowed, the geometry must stay unchanged\n";
    }
    if (m_pUseMagneticField) {
      bool magChanged = idealMagRcdWatcher_.check(iSetup);
      if (magChanged && (!firstRun)) {
        throw cms::Exception("BadConfig") 
          << "[SimG4Core RunManager]\n"
          << "The MagneticField configuration is changed during the job execution\n"
          << "this is not allowed, the MagneticField must stay unchanged\n";
      }
    }
    firstRun = false;
  });

  ESProducts ret;

  // DDDWorld: get the DDCV from the ES and use it to build the World
  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(pDD);
  ret.pDD = pDD.product();

  if(m_pUseMagneticField) {
    edm::ESHandle<MagneticField> pMF;
    iSetup.get<IdealMagneticFieldRecord>().get(pMF);
    ret.pMF = pMF.product();
  }

  edm::ESHandle<HepPDT::ParticleDataTable> fTable;
  iSetup.get<PDTRecord>().get(fTable);
  ret.pTable = fTable.product();
  return ret;
}
