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
// $Id: SimTrackManager.h,v 1.2 2005/11/28 00:43:10 chrjones Exp $
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
  //      enum SpecialNumbers {InvalidID = 65535};
      /// this map contains association between vertex number and position
      typedef std::pair<int,Hep3Vector> MapVertexPosition;
      typedef std::vector<std::pair<int,Hep3Vector> > MapVertexPositionVector;
      typedef std::map<int,MapVertexPositionVector> MotherParticleToVertexMap;
      typedef MotherParticleToVertexMap VertexMap;
     
      SimTrackManager(bool iCollapsePrimaryVertices =false);
      virtual ~SimTrackManager();

      // ---------- const member functions ---------------------
      const TrackContainer * trackContainer() const { 
	return m_trksForThisEvent; }


      // ---------- member functions ---------------------------
      void storeTracks(G4SimEvent * simEvent);

      void reset();
      void deleteTracks();

      void addTrack(TrackWithHistory* iTrack) {
	m_trksForThisEvent->push_back(iTrack);
      }

      void setCollapsePrimaryVertices(bool iSet) {
	m_collapsePrimaryVertices=iSet;
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
      MotherParticleToVertexMap m_vertexMap;
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
