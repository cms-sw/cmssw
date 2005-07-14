#ifndef SimG4Core_EventAction_H
#define SimG4Core_EventAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h" 

#include "G4UserEventAction.hh"

#include <vector>
#include <map>
 
class RunManager;
 
class EventAction: public G4UserEventAction
{
public:
    enum SpecialNumbers {InvalidID = 65535};
    /// this map contains association between vertex number and position
    typedef std::pair<int,Hep3Vector> MapVertexPosition;
    typedef std::vector<std::pair<int,Hep3Vector> > MapVertexPositionVector;
    typedef std::map<int,MapVertexPositionVector> MotherParticleToVertexMap;
    typedef MotherParticleToVertexMap VertexMap;
    typedef std::map<unsigned int,unsigned int> G4ToSimMapType;
    typedef std::vector<unsigned int> SimToG4VectorType;
public:
    //EventAction(const edm::ParameterSet & ps);
    EventAction();
    ~EventAction();
    void BeginOfEventAction(const G4Event * evt);
    void EndOfEventAction(const G4Event * evt);
    void saveTrackAndItsBranch(int i);
    void storeTracks(G4SimEvent * simEvent);
    unsigned int g4ToSim(unsigned int) const;
    unsigned int simToG4(unsigned int) const;
    void cleanVertexMap();
    int getOrCreateVertex(TrackWithHistory *,int,G4SimEvent * simEvent);
    TrackContainer * trackContainer() { return m_trksForThisEvent; }
private:
    TrackContainer * m_trksForThisEvent;
    bool m_SaveSimTracks;
    G4ToSimMapType m_g4ToSimMap;
    MotherParticleToVertexMap m_vertexMap;
    SimToG4VectorType m_simToG4Vector;
    int m_nVertices;
    bool m_collapsePrimaryVertices;
    std::string m_stopFile;
    bool m_debug;
};

class trkIDLess
{
public:
    bool operator()(TrackWithHistory * trk1, TrackWithHistory * trk2) const
    { return (trk1->trackID() < trk2->trackID()); }
};

#endif



