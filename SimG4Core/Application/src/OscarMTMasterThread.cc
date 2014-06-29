#include "SimG4Core/Application/interface/OscarMTMasterThread.h"

#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "G4PhysicalVolumeStore.hh"



OscarMTMasterThread::OscarMTMasterThread(const RunManagerMTInit *runManagerInit):
  m_runManagerInit(runManagerInit),
  m_masterThreadState(ThreadState::NotExist),
  m_masterCanProceed(false),
  m_mainCanProceed(false),
  m_stopped(false)
{

  const edm::ParameterSet& pset = m_runManagerInit->parameterSet();

  // Lock the mutex
  //std::unique_lock<std::mutex> lk(m_notifyMasterMutex);
  std::unique_lock<std::mutex> lk(m_threadMutex);

  LogDebug("OscarMTMasterThread") << "Master thread: Creating master thread";

  // Create Genat4 master thread
  m_masterThread = std::thread([&](){
      /////////////////
      // Initialization

      std::shared_ptr<RunManagerMT> runManagerMaster;
      std::unique_ptr<CustomUIsession> uiSession;

      // Lock the mutex (i.e. wait until the creating thread has called cv.wait()
      std::unique_lock<std::mutex> lk2(m_threadMutex);

      LogDebug("OscarMTMasterThread") << "Main thread: initializing RunManagerMT";

      //UIsession manager for message handling
      uiSession.reset(new CustomUIsession());

      // Create the master run manager, and share it to the CMSSW thread
      runManagerMaster = std::make_shared<RunManagerMT>(pset);
      m_runManagerMaster = runManagerMaster;

      LogDebug("OscarMTMasterThread") << "Master thread: RunManagerMT initialization finished";

      /////////////
      // State loop
      bool isG4Alive = false;
      while(true) {
        // Signal main thread that it can proceed
        m_mainCanProceed = true;
        LogDebug("OscarMTMasterThread") << "Master thread: State loop, notify main thread";
        m_notifyMainCv.notify_one();

        // Wait until the main thread sends signal
        m_masterCanProceed = false;
        LogDebug("OscarMTMasterThread") << "Master thread: State loop, starting wait";
        m_notifyMasterCv.wait(lk2, [&]{return m_masterCanProceed;});

        // Act according to the state
        LogDebug("OscarMTMasterThread") << "Master thread: Woke up, state is " << static_cast<int>(m_masterThreadState);
        if(m_masterThreadState == ThreadState::BeginRun) {
          // Initialize Geant4
          LogDebug("OscarMTMasterThread") << "Master thread: Initializing Geant4";
          runManagerMaster->initG4(m_esProducts.pDD, m_esProducts.pMF, m_esProducts.pTable);
          isG4Alive = true;
        }
        else if(m_masterThreadState == ThreadState::EndRun) {
          // Stop Geant4
          LogDebug("OscarMTMasterThread") << "Master thread: Stopping Geant4";
          runManagerMaster->stopG4();
          isG4Alive = false;
        }
        else if(m_masterThreadState == ThreadState::Destruct) {
          LogDebug("OscarMTMasterThread") << "Master thread: Breaking out of state loop";
          if(isG4Alive)
            throw cms::Exception("Assert") << "Geant4 is still alive, master thread state must be set to EndRun before Destruct";
          break;
        }
        else {
          throw cms::Exception("Assert") << "Illegal master thread state " << static_cast<int>(m_masterThreadState);
        }
      }

      //////////
      // Cleanup
      LogDebug("OscarMTMasterThread") << "Master thread: Am I unique owner of runManagerMaster? " << runManagerMaster.unique();

      // must be done in this thread, segfault otherwise
      runManagerMaster.reset();
      G4PhysicalVolumeStore::Clean();

      LogDebug("OscarMTMasterThread") << "Master thread: Reseted shared_ptr";
      lk2.unlock();
      LogDebug("OscarMTMasterThread") << "Master thread: Finished";
    });

  // Start waiting a signal from the condition variable (releases the lock temporarily)
  // First for initialization
  m_mainCanProceed = false;
  LogDebug("OscarMTMasterThread") << "Main thread: Signal master for initialization";
  m_notifyMainCv.wait(lk, [&](){return m_mainCanProceed;});

  lk.unlock();
  LogDebug("OscarMTMasterThread") << "Main thread: Finish constructor";
}

OscarMTMasterThread::~OscarMTMasterThread() {
  if(!m_stopped) {
    edm::LogError("OscarMTMasterThread") << "OscarMTMasterThread::stopThread() has not been called to stop Geant4 and join the master thread";
  }
}

void OscarMTMasterThread::beginRun(const edm::EventSetup& iSetup) const {
  std::lock_guard<std::mutex> lk(m_protectMutex);

  std::unique_lock<std::mutex> lk2(m_threadMutex);

  // Reading from ES must be done in the main (CMSSW) thread
  m_esProducts = m_runManagerInit->readES(iSetup);

  m_masterThreadState = ThreadState::BeginRun;
  m_masterCanProceed = true;
  m_mainCanProceed = false;
  LogDebug("OscarMTMasterThread") << "Main thread: Signal master for BeginRun";
  m_notifyMasterCv.notify_one();
  m_notifyMainCv.wait(lk2, [&](){return m_mainCanProceed;});

  m_esProducts.reset();
  lk2.unlock();
  LogDebug("OscarMTMasterThread") << "Main thread: Finish beginRun";
}

void OscarMTMasterThread::endRun() const {
  std::lock_guard<std::mutex> lk(m_protectMutex);

  std::unique_lock<std::mutex> lk2(m_threadMutex);
  m_masterThreadState = ThreadState::EndRun;
  m_mainCanProceed = false;
  m_masterCanProceed = true;
  LogDebug("OscarMTMasterThread") << "Main thread: signal master thread for EndRun";
  m_notifyMasterCv.notify_one();
  m_notifyMainCv.wait(lk2, [&](){return m_mainCanProceed;});
  lk2.unlock();
  LogDebug("OscarMTMasterThread") << "Main thread: Finish endRun";
}

void OscarMTMasterThread::stopThread() {
  if(m_stopped) {
    edm::LogError("OscarMTMasterThread") << "Second call to OscarMTMasterThread::stopThread(), not doing anything";
    return;
  }
  LogDebug("OscarMTMasterThread") << "Main thread: stopThread()";

  // Release our instance of the shared master run manager, so that
  // the G4 master thread can do the cleanup. Then notify the master
  // thread, and join it.
  std::unique_lock<std::mutex> lk2(m_threadMutex);
  m_runManagerMaster.reset();
  LogDebug("OscarMTMasterThread") << "Main thread: reseted shared_ptr";

  m_masterThreadState = ThreadState::Destruct;
  m_masterCanProceed = true;
  LogDebug("OscarMTMasterThread") << "Main thread: signal master thread for Destruct";
  m_notifyMasterCv.notify_one();
  lk2.unlock();

  LogDebug("OscarMTMasterThread") << "Main thread: joining master thread";
  m_masterThread.join();
  LogDebug("OscarMTMasterThread") << "Main thread: finished";
  m_stopped = true;
}

