#include "SimG4Core/Application/interface/OscarMTMasterThread.h"

#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/CustomUIsession.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

#include "G4PhysicalVolumeStore.hh"

OscarMTMasterThread::OscarMTMasterThread(const edm::ParameterSet& iConfig):
  m_pUseMagneticField(iConfig.getParameter<bool>("UseMagneticField")),
  m_pDD(nullptr), m_pMF(nullptr), m_pTable(nullptr),
  m_masterThreadState(ThreadState::NotExist),
  m_masterCanProceed(false),
  m_mainCanProceed(false),
  m_firstRun(true),
  m_stopped(false)
{
  // Lock the mutex
  std::unique_lock<std::mutex> lk(m_threadMutex);

  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread: creating master thread";

  // Create Genat4 master thread
  m_masterThread = std::thread([&](){
      /////////////////
      // Initialization

      std::shared_ptr<RunManagerMT> runManagerMaster;
      std::unique_ptr<CustomUIsession> uiSession;

      // Lock the mutex (i.e. wait until the creating thread has called cv.wait()
      std::unique_lock<std::mutex> lk2(m_threadMutex);

      edm::LogInfo("SimG4CoreApplication") 
        << "OscarMTMasterThread: initializing RunManagerMT";

      //UIsession manager for message handling
      uiSession.reset(new CustomUIsession());

      // Create the master run manager, and share it to the CMSSW thread
      runManagerMaster = std::make_shared<RunManagerMT>(iConfig);
      m_runManagerMaster = runManagerMaster;

      edm::LogInfo("SimG4CoreApplication") 
        << "OscarMTMasterThread: initialization of RunManagerMT finished";

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
          runManagerMaster->initG4(m_pDD, m_pMF, m_pTable);
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
            throw edm::Exception(edm::errors::LogicError) 
	      << "Geant4 is still alive, master thread state must be set to EndRun before Destruct";
          break;
        }
        else {
          throw edm::Exception(edm::errors::LogicError) 
	    << "OscarMTMasterThread: Illegal master thread state " 
	    << static_cast<int>(m_masterThreadState);
        }
      }

      //////////
      // Cleanup
      edm::LogInfo("SimG4CoreApplication") 
        << "OscarMTMasterThread: start RunManagerMT destruction";
      LogDebug("OscarMTMasterThread") 
      << "Master thread: Am I unique owner of runManagerMaster? " 
      << runManagerMaster.unique();

      // must be done in this thread, segfault otherwise
      runManagerMaster.reset();
      G4PhysicalVolumeStore::Clean();

      LogDebug("OscarMTMasterThread") << "Master thread: Reseted shared_ptr";
      lk2.unlock();
      edm::LogInfo("SimG4CoreApplication") 
        << "OscarMTMasterThread: Master thread is finished";
    });

  // Start waiting a signal from the condition variable (releases the lock temporarily)
  // First for initialization
  m_mainCanProceed = false;
  LogDebug("OscarMTMasterThread") << "Main thread: Signal master for initialization";
  m_notifyMainCv.wait(lk, [&](){return m_mainCanProceed;});

  lk.unlock();
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread: Master thread is constructed";
}

OscarMTMasterThread::~OscarMTMasterThread() {
  if(!m_stopped) {
    stopThread();  
  }
}

void OscarMTMasterThread::beginRun(const edm::EventSetup& iSetup) const {
  std::lock_guard<std::mutex> lk(m_protectMutex);

  std::unique_lock<std::mutex> lk2(m_threadMutex);

  // Reading from ES must be done in the main (CMSSW) thread
  readES(iSetup);

  m_masterThreadState = ThreadState::BeginRun;
  m_masterCanProceed = true;
  m_mainCanProceed = false;
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread: Signal master for BeginRun";
  m_notifyMasterCv.notify_one();
  m_notifyMainCv.wait(lk2, [&](){return m_mainCanProceed;});

  lk2.unlock();
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread: finish BeginRun";
}

void OscarMTMasterThread::endRun() const {
  std::lock_guard<std::mutex> lk(m_protectMutex);

  std::unique_lock<std::mutex> lk2(m_threadMutex);
  m_masterThreadState = ThreadState::EndRun;
  m_mainCanProceed = false;
  m_masterCanProceed = true;
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread: Signal master for EndRun";
  m_notifyMasterCv.notify_one();
  m_notifyMainCv.wait(lk2, [&](){return m_mainCanProceed;});
  lk2.unlock();
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread: finish EndRun";
}

void OscarMTMasterThread::stopThread() {
  if(m_stopped) {
    return;
  }
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread::stopTread: stop main thread";

  // Release our instance of the shared master run manager, so that
  // the G4 master thread can do the cleanup. Then notify the master
  // thread, and join it.
  std::unique_lock<std::mutex> lk2(m_threadMutex);
  m_runManagerMaster.reset();
  LogDebug("OscarMTMasterThread") << "Main thread: reseted shared_ptr";

  m_masterThreadState = ThreadState::Destruct;
  m_masterCanProceed = true;
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread::stopTread: stop main thread";
  m_notifyMasterCv.notify_one();
  lk2.unlock();

  LogDebug("OscarMTMasterThread") << "Main thread: joining master thread";
  m_masterThread.join();
  edm::LogInfo("SimG4CoreApplication") 
    << "OscarMTMasterThread::stopTread: main thread finished";
  m_stopped = true;
}

void OscarMTMasterThread::readES(const edm::EventSetup& iSetup) const {
  bool geomChanged = idealGeomRcdWatcher_.check(iSetup);
  if (geomChanged && (!m_firstRun)) {
    throw edm::Exception(edm::errors::Configuration)
      << "[SimG4Core OscarMTMasterThread]\n"
      << "The Geometry configuration is changed during the job execution\n"
      << "this is not allowed, the geometry must stay unchanged";
  }
  if (m_pUseMagneticField) {
    bool magChanged = idealMagRcdWatcher_.check(iSetup);
    if (magChanged && (!m_firstRun)) {
      throw edm::Exception(edm::errors::Configuration)
        << "[SimG4Core OscarMTMasterThread]\n"
	<< "The MagneticField configuration is changed during the job execution\n"
	<< "this is not allowed, the MagneticField must stay unchanged";
    }
  }
  // Don't read from ES if not the first run, just as in
  if(!m_firstRun)
    return;

  // DDDWorld: get the DDCV from the ES and use it to build the World
  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(pDD);
  m_pDD = pDD.product();

  if(m_pUseMagneticField) {
    edm::ESHandle<MagneticField> pMF;
    iSetup.get<IdealMagneticFieldRecord>().get(pMF);
    m_pMF = pMF.product();
  }

  edm::ESHandle<HepPDT::ParticleDataTable> fTable;
  iSetup.get<PDTRecord>().get(fTable);
  m_pTable = fTable.product();

  m_firstRun = false;
}
