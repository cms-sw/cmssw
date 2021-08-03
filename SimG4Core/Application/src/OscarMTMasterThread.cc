#include <memory>

#include "SimG4Core/Application/interface/OscarMTMasterThread.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Geometry/interface/CustomUIsession.h"

#include "G4PhysicalVolumeStore.hh"

OscarMTMasterThread::OscarMTMasterThread(const edm::ParameterSet& iConfig)
    : m_pGeoFromDD4hep(iConfig.getParameter<bool>("g4GeometryDD4hepSource")),
      m_masterThreadState(ThreadState::NotExist) {
  // Lock the mutex
  std::unique_lock<std::mutex> lk(m_threadMutex);

  edm::LogVerbatim("SimG4CoreApplication")
      << "OscarMTMasterThread: creating master thread DD4Hep: " << m_pGeoFromDD4hep;

  // Create Geant4 master thread
  m_masterThread = std::thread([&]() {
    /////////////////
    // Initialization
    std::unique_ptr<CustomUIsession> uiSession;

    // Lock the mutex (i.e. wait until the creating thread has called cv.wait()
    std::unique_lock<std::mutex> lk2(m_threadMutex);

    edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: initializing RunManagerMT";

    //UIsession manager for message handling
    uiSession = std::make_unique<CustomUIsession>();

    // Create the master run manager, and share it to the CMSSW thread
    m_runManagerMaster = std::make_shared<RunManagerMT>(iConfig);

    /////////////
    // State loop
    bool isG4Alive = false;
    while (true) {
      // Signal main thread that it can proceed
      m_mainCanProceed = true;
      edm::LogVerbatim("OscarMTMasterThread") << "Master thread: State loop, notify main thread";
      m_notifyMainCv.notify_one();

      // Wait until the main thread sends signal
      m_masterCanProceed = false;
      edm::LogVerbatim("OscarMTMasterThread") << "Master thread: State loop, starting wait";
      m_notifyMasterCv.wait(lk2, [&] { return m_masterCanProceed; });
      //m_notifyMasterCv.wait(lk2, [&] { return false; });

      // Act according to the state
      edm::LogVerbatim("OscarMTMasterThread")
          << "Master thread: Woke up, state is " << static_cast<int>(m_masterThreadState);
      if (m_masterThreadState == ThreadState::BeginRun) {
        // Initialize Geant4
        edm::LogVerbatim("OscarMTMasterThread") << "Master thread: Initializing Geant4";
        m_runManagerMaster->initG4(m_pDDD, m_pDD4Hep, m_pTable);
        isG4Alive = true;
      } else if (m_masterThreadState == ThreadState::EndRun) {
        // Stop Geant4
        edm::LogVerbatim("OscarMTMasterThread") << "Master thread: Stopping Geant4";
        m_runManagerMaster->stopG4();
        isG4Alive = false;
      } else if (m_masterThreadState == ThreadState::Destruct) {
        edm::LogVerbatim("OscarMTMasterThread") << "Master thread: Breaking out of state loop";
        if (isG4Alive)
          throw edm::Exception(edm::errors::LogicError)
              << "Geant4 is still alive, master thread state must be set to EndRun before Destruct";
        break;
      } else {
        throw edm::Exception(edm::errors::LogicError)
            << "OscarMTMasterThread: Illegal master thread state " << static_cast<int>(m_masterThreadState);
      }
    }

    //////////
    // Cleanup
    edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: start RunManagerMT destruction";

    // must be done in this thread, segfault otherwise
    m_runManagerMaster.reset();
    G4PhysicalVolumeStore::Clean();

    edm::LogVerbatim("OscarMTMasterThread") << "Master thread: Reseted shared_ptr";
    lk2.unlock();
    edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: Master thread is finished";
  });

  // Start waiting a signal from the condition variable (releases the lock temporarily)
  // First for initialization
  m_mainCanProceed = false;
  LogDebug("OscarMTMasterThread") << "Main thread: Signal master for initialization";
  m_notifyMainCv.wait(lk, [&]() { return m_mainCanProceed; });

  lk.unlock();
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: Master thread is constructed";
}

OscarMTMasterThread::~OscarMTMasterThread() {
  if (!m_stopped) {
    stopThread();
  }
}

void OscarMTMasterThread::callConsumes(edm::ConsumesCollector&& iC) const {
  if (m_hasToken) {
    return;
  }
  if (m_pGeoFromDD4hep) {
    m_DD4Hep = iC.esConsumes<cms::DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
  } else {
    m_DDD = iC.esConsumes<DDCompactView, IdealGeometryRecord, edm::Transition::BeginRun>();
  }
  m_PDT = iC.esConsumes<HepPDT::ParticleDataTable, PDTRecord, edm::Transition::BeginRun>();
  m_hasToken = true;
}

void OscarMTMasterThread::beginRun(const edm::EventSetup& iSetup) const {
  std::lock_guard<std::mutex> lk(m_protectMutex);
  std::unique_lock<std::mutex> lk2(m_threadMutex);
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread::beginRun";

  if (m_firstRun) {
    if (m_pGeoFromDD4hep) {
      m_pDD4Hep = &(*iSetup.getTransientHandle(m_DD4Hep));
    } else {
      m_pDDD = &(*iSetup.getTransientHandle(m_DDD));
    }
    m_pTable = &iSetup.getData(m_PDT);
    m_firstRun = false;
  }
  m_masterThreadState = ThreadState::BeginRun;
  m_masterCanProceed = true;
  m_mainCanProceed = false;
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: Signal master for BeginRun";
  m_notifyMasterCv.notify_one();
  m_notifyMainCv.wait(lk2, [&]() { return m_mainCanProceed; });

  lk2.unlock();
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: finish BeginRun";
}

void OscarMTMasterThread::endRun() const {
  std::lock_guard<std::mutex> lk(m_protectMutex);
  std::unique_lock<std::mutex> lk2(m_threadMutex);

  m_masterThreadState = ThreadState::EndRun;
  m_mainCanProceed = false;
  m_masterCanProceed = true;
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: Signal master for EndRun";
  m_notifyMasterCv.notify_one();
  m_notifyMainCv.wait(lk2, [&]() { return m_mainCanProceed; });
  lk2.unlock();
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread: finish EndRun";
}

void OscarMTMasterThread::stopThread() {
  if (m_stopped) {
    return;
  }
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread::stopTread: stop main thread";

  // Release our instance of the shared master run manager, so that
  // the G4 master thread can do the cleanup. Then notify the master
  // thread, and join it.
  std::unique_lock<std::mutex> lk2(m_threadMutex);
  m_runManagerMaster.reset();
  LogDebug("OscarMTMasterThread") << "Main thread: reseted shared_ptr";

  m_masterThreadState = ThreadState::Destruct;
  m_masterCanProceed = true;
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread::stopTread: notify";
  m_notifyMasterCv.notify_one();
  lk2.unlock();

  LogDebug("OscarMTMasterThread") << "Main thread: joining master thread";
  m_masterThread.join();
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTMasterThread::stopTread: main thread finished";
  m_stopped = true;
}
