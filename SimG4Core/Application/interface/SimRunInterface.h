#ifndef SimG4Core_SimRunInterface_h
#define SimG4Core_SimRunInterface_h 1

// This class is needed to provide an interface
// between Geant4 user actions and CMS SIM
// infrastructure both for sequentional and MT runs


class RunManager;
class RunManagerMT;
class RunManagerMTWorker;
class SimTrackManager;
class RunAction;
class EventAction;
class TrackingAction;
class SteppingAction;
class G4SimEvent;

class SimRunInterface
{
public:

  SimRunInterface(RunManager* run, bool master);

  SimRunInterface(RunManagerMT* run, bool master);

  SimRunInterface(RunManagerMTWorker* run, bool master);

  ~SimRunInterface();

  // Needed because for workers SumRunInterface sits in TLS, while
  // RunManagerMTWorkers are members of edm::stream OscarMTProducer
  void setRunManagerMTWorker(RunManagerMTWorker *run);

  void Connect(RunAction*);

  void Connect(EventAction*);

  void Connect(TrackingAction*);

  void Connect(SteppingAction*);

  SimTrackManager* GetSimTrackManager();

  void abortEvent();

  void abortRun(bool softAbort);

  G4SimEvent* simEvent();

private:

  RunManager* m_runManager;
  RunManagerMT* m_runManagerMT;
  RunManagerMTWorker *m_runManagerMTWorker;

  SimTrackManager* m_SimTrackManager;

  bool  m_isMaster;             
};

#endif

    
