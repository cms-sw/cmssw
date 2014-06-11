#include "SimG4Core/Application/interface/OscarMTMasterThread.h"

#include "SimG4Core/Application/interface/RunManagerMTInit.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4PhysicalVolumeStore.hh"



OscarMTMasterThread::OscarMTMasterThread(std::shared_ptr<RunManagerMTInit> runManagerInit, const edm::EventSetup& iSetup):
  m_runManagerInit(runManagerInit)
{

  const edm::ParameterSet& pset = m_runManagerInit->parameterSet();
  SimActivityRegistry *registry = m_runManagerInit->registry(); // must be done in the CMSSW thread
  RunManagerMTInit::ESProducts esprod = m_runManagerInit->readES(iSetup);

  // Lock the mutex
  std::unique_lock<std::mutex> lk(m_startMutex);

  // Create Genat4 master thread
  m_masterThread = std::thread([&](){
      std::shared_ptr<RunManagerMT> runManagerMaster;
      std::unique_ptr<CustomUIsession> uiSession;
      {
        // Lock the mutex (i.e. wait until the creating thread has called cv.wait()
        std::lock_guard<std::mutex> lk2(m_startMutex);

        // Create the master run manager, and share it to the CMSSW thread
        runManagerMaster = std::make_shared<RunManagerMT>(pset, registry);
        m_runManagerMaster = runManagerMaster;

        //UIsession manager for message handling
        uiSession.reset(new CustomUIsession());

        // Initialize Geant4
        runManagerMaster->initG4(esprod.pDD, esprod.pMF, esprod.pTable, iSetup);
      }
      // G4 initialization finish, send signal to the other thread to continue
      m_startCanProceed = true;
      m_startCv.notify_one();
      //edm::LogWarning("Test") << "Master thread, notified main thread";

      // Lock the other mutex, and wait a signal via the condition variable
      std::unique_lock<std::mutex> lk2(m_stopMutex);
      //edm::LogWarning("Test") << "Master thread, locked mutex, starting wait";
      m_stopCanProceed = false;
      m_stopCv.wait(lk2, [&](){return m_stopCanProceed;});

      // After getting a signal from the CMSSW thread, do clean-up
      //edm::LogWarning("Test") << "Master thread, woke up, starting cleanup";
      runManagerMaster->stopG4();

      //edm::LogWarning("Test") << "Master thread, stopped G4, am I unique owner? " << runManagerMaster.unique();

      // must be done in this thread, segfault otherwise
      runManagerMaster.reset();
      G4PhysicalVolumeStore::Clean();

      //edm::LogWarning("Test") << "Master thread, reseted shared_ptr";
      lk2.unlock();
      //edm::LogWarning("Test") << "Master thread, finished";
    });

  // Start waiting a signal from the condition variable (releases the lock temporarily)
  m_startCanProceed = false;
  m_startCv.wait(lk, [&](){return m_startCanProceed;});
  // Unlock the lock
  lk.unlock();
  //edm::LogWarning("Test") << "Main thread, again address " << esprod.pDD;
}

OscarMTMasterThread::~OscarMTMasterThread() {
  //edm::LogWarning("Test") << "Main thread, destructor";
  // Release our instance of the shared master run manager, so that
  // the G4 master thread can do the cleanup. Then notify the master
  // thread, and join it.
  {
    std::lock_guard<std::mutex> lk(m_stopMutex);
    m_runManagerMaster.reset();
    //edm::LogWarning("Test") << "Main thread, reseted shared_ptr";
  }
  //edm::LogWarning("Test") << "Main thread, going to signal master thread";
  m_stopCanProceed = true;
  m_stopCv.notify_one();
  //edm::LogWarning("Test") << "Main thread, going to join master thread";
  m_masterThread.join();
  //edm::LogWarning("Test") << "Main thread, finished";
}
