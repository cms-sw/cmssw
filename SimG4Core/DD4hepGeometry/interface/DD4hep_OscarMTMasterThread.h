#ifndef SIM_G4_CORE_DD4HEP_OSCAR_MT_MASTER_THREAD_H
#define SIM_G4_CORE_DD4HEP_OSCAR_MT_MASTER_THREAD_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace edm {
  class EventSetup;
}

class DD4hep_RunManagerMT;

namespace cms {
  class DDDetector;
  struct DDSpecParRegistry;
}

class MagneticField;

namespace HepPDT {
  class ParticleDataTable;
}

class DD4hep_OscarMTMasterThread {
public:
  explicit DD4hep_OscarMTMasterThread(const edm::ParameterSet& iConfig);
  ~DD4hep_OscarMTMasterThread();

  void beginRun(const edm::EventSetup& iSetup) const;
  void endRun() const;
  void stopThread();

  inline DD4hep_RunManagerMT& runManagerMaster() const { return *m_runManagerMaster; }
  inline DD4hep_RunManagerMT *runManagerMasterPtr() const { return m_runManagerMaster.get(); }

private:
  void readES(const edm::EventSetup& iSetup) const;

  enum class ThreadState {
    NotExist=0, BeginRun=1, EndRun=2, Destruct=3
  };

  const bool m_pUseMagneticField;

  std::shared_ptr<DD4hep_RunManagerMT> m_runManagerMaster;
  std::thread m_masterThread;

  // ES products needed for Geant4 initialization
  mutable edm::ESWatcher<GeometryFileRcd> idealGeomRcdWatcher_;
  mutable edm::ESWatcher<IdealMagneticFieldRecord> idealMagRcdWatcher_;
  mutable const cms::DDDetector *m_pDD;
  mutable const cms::DDSpecParRegistry *m_registry;
  mutable const MagneticField *m_pMF;
  mutable const HepPDT::ParticleDataTable *m_pTable;

  mutable std::mutex m_protectMutex;
  mutable std::mutex m_threadMutex;
  mutable std::condition_variable m_notifyMasterCv;
  mutable std::condition_variable m_notifyMainCv;

  mutable ThreadState m_masterThreadState;

  mutable bool m_masterCanProceed;
  mutable bool m_mainCanProceed;
  mutable bool m_firstRun;
  mutable bool m_stopped;
};

#endif
