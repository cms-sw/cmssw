#ifndef SimG4Core_OscarMTMasterThread_H
#define SimG4Core_OscarMTMasterThread_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace edm {
  class EventSetup;
}

class RunManagerMT;

class DDCompactView;
class MagneticField;

namespace HepPDT {
  class ParticleDataTable;
}

class OscarMTMasterThread {
public:
  explicit OscarMTMasterThread(const edm::ParameterSet& iConfig);
  ~OscarMTMasterThread();

  void beginRun(const edm::EventSetup& iSetup) const;
  void endRun() const;
  void stopThread();

  inline RunManagerMT& runManagerMaster() const { return *m_runManagerMaster; }
  inline RunManagerMT *runManagerMasterPtr() const { return m_runManagerMaster.get(); }

private:
  void readES(const edm::EventSetup& iSetup) const;

  enum class ThreadState {
    NotExist=0, BeginRun=1, EndRun=2, Destruct=3
  };

  const bool m_pUseMagneticField;

  std::shared_ptr<RunManagerMT> m_runManagerMaster;
  std::thread m_masterThread;

  // ES products needed for Geant4 initialization
  mutable edm::ESWatcher<IdealGeometryRecord> idealGeomRcdWatcher_;
  mutable edm::ESWatcher<IdealMagneticFieldRecord> idealMagRcdWatcher_;
  mutable const DDCompactView *m_pDD;
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
