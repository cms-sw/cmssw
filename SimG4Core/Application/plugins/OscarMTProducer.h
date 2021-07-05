#ifndef SimG4Core_OscarMTProducer_H
#define SimG4Core_OscarMTProducer_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "SimG4Core/Application/interface/OscarMTMasterThread.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "G4Threading.hh"

#include <memory>

class SimProducer;
class RunManagerMTWorker;

class OscarMTProducer : public edm::stream::EDProducer<edm::GlobalCache<OscarMTMasterThread>,
                                                       edm::RunCache<int>  // for some reason void doesn't compile
                                                       > {
public:
  typedef std::vector<std::shared_ptr<SimProducer> > Producers;

  explicit OscarMTProducer(edm::ParameterSet const& p, const OscarMTMasterThread*);
  ~OscarMTProducer() override;

  static std::unique_ptr<OscarMTMasterThread> initializeGlobalCache(const edm::ParameterSet& iConfig);
  static std::shared_ptr<int> globalBeginRun(const edm::Run& iRun,
                                             const edm::EventSetup& iSetup,
                                             const OscarMTMasterThread* masterThread);
  static void globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext* iContext);
  static void globalEndJob(OscarMTMasterThread* masterThread);

  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  std::unique_ptr<RunManagerMTWorker> m_runManagerWorker;
  const OscarMTMasterThread* m_masterThread = nullptr;

  static edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> m_DD4Hep;
  static edm::ESGetToken<DDCompactView, IdealGeometryRecord> m_DDD;
  static edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> m_PDT;
  static edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_MagField;
  static G4Mutex m_OscarMutex;
  static bool m_hasToken;
};

#endif
