#ifndef SimG4CMS_TkAccumulatingSensitiveDetector_H
#define SimG4CMS_TkAccumulatingSensitiveDetector_H

/**
 * This is TkSensitiveDetector which accumulates all the SimHits coming from the same track in the
 * same volume, thus reducing the db size.
 */
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"

#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"

#include <string>

class TrackInformation;
class SimTrackManager;
class TrackingSlaveSD;
class FrameRotation;
class UpdatablePSimHit;
class G4ProcessTypeEnumerator;
class G4TrackToParticleID;
class TrackerG4SimHitNumberingScheme;

class TkAccumulatingSensitiveDetector : 
public SensitiveTkDetector, 
public Observer<const BeginOfEvent*>,
public Observer<const BeginOfTrack*>,
public Observer<const BeginOfJob*>
{ 
public:    
    TkAccumulatingSensitiveDetector(const std::string&, const DDCompactView &,
				    const SensitiveDetectorCatalog &,
				    edm::ParameterSet const &,
				    const SimTrackManager*);
    ~TkAccumulatingSensitiveDetector() override;
    bool ProcessHits(G4Step *,G4TouchableHistory *) override;
    uint32_t setDetUnitId(const G4Step*) override;
    void EndOfEvent(G4HCofThisEvent*) override;

    void fillHits(edm::PSimHitContainer&, const std::string& use) override;
    std::vector<std::string> getNames() override;

private:
    virtual void sendHit();
    virtual void updateHit(const G4Step *);
    virtual bool newHit(const G4Step *);
    virtual bool closeHit(const G4Step *);
    virtual void createHit(const G4Step *);
    void checkExitPoint(const Local3DPoint&);
    void update(const BeginOfEvent *) override;
    void update(const BeginOfTrack *) override;
    void update(const BeginOfJob *) override;
    void clearHits() override;
    Local3DPoint toOrcaRef(const Local3DPoint& ,const G4VPhysicalVolume *);
    int tofBin(float);
   
    TrackingSlaveSD * slaveLowTof;
    TrackingSlaveSD * slaveHighTof;
    FrameRotation * myRotation;
    UpdatablePSimHit * mySimHit;
    std::string pname;
    Local3DPoint globalEntryPoint;
    Local3DPoint globalExitPoint;
    const SimTrackManager* theManager;
    const G4VPhysicalVolume * oldVolume;
    G4ProcessTypeEnumerator * theG4ProcessTypeEnumerator;
    double theSigma;
    uint32_t lastId;
    unsigned int lastTrack;
    int eventno;
    // cache stuff for debugging
    float px,py,pz;
    bool allowZeroEnergyLoss;
    bool printHits;
    bool neverAccumulate;
    G4TrackToParticleID * myG4TrackToParticleID;
    TrackInformation* getOrCreateTrackInformation(const G4Track *);
    float energyCut;
    float energyHistoryCut;
    //
    // definition of Tracker volume
    //
    float rTracker;
    float zTracker;

    TrackerG4SimHitNumberingScheme* numberingScheme_;
};

#endif





