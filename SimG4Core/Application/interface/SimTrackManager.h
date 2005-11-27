#ifndef Application_SimTrackManager_h
#define Application_SimTrackManager_h
// -*- C++ -*-
//
// Package:     Application
// Class  :     SimTrackManager
// 
/**\class SimTrackManager SimTrackManager.h SimG4Core/Application/interface/SimTrackManager.h

 Description: Holds tracking information used by the sensitive detectors

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Fri Nov 25 17:36:41 EST 2005
// $Id$
//

// system include files
#include <map>
#include <vector>

// user include files
#include "SimG4Core/Application/interface/G4SimEvent.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/TrackContainer.h" 

// forward declarations

class SimTrackManager
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
     
      SimTrackManager(bool iCollapsePrimaryVertices);
      virtual ~SimTrackManager();

      // ---------- const member functions ---------------------
      unsigned int g4ToSim(unsigned int) const;
      unsigned int simToG4(unsigned int) const;
      const TrackContainer * trackContainer() const { 
	return m_trksForThisEvent; }


      // ---------- member functions ---------------------------
      void storeTracks(G4SimEvent * simEvent);

      void reset();
      void deleteTracks();

      void addTrack(TrackWithHistory* iTrack) {
	m_trksForThisEvent->push_back(iTrack);
      }
   private:
      SimTrackManager(const SimTrackManager&); // stop default

      const SimTrackManager& operator=(const SimTrackManager&); // stop default

      void saveTrackAndItsBranch(int i);
      int getOrCreateVertex(TrackWithHistory *,int,G4SimEvent * simEvent);
      void cleanVertexMap();
      void reallyStoreTracks(G4SimEvent * simEvent);

      // ---------- member data --------------------------------
      TrackContainer * m_trksForThisEvent;
      bool m_SaveSimTracks;
      G4ToSimMapType m_g4ToSimMap;
      MotherParticleToVertexMap m_vertexMap;
      SimToG4VectorType m_simToG4Vector;
      int m_nVertices;
      bool m_collapsePrimaryVertices;
};


class trkIDLess
{
public:
    bool operator()(TrackWithHistory * trk1, TrackWithHistory * trk2) const
    { return (trk1->trackID() < trk2->trackID()); }
};

#endif
