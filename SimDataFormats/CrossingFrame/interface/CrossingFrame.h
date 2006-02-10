#ifndef CROSSING_FRAME_H
#define CROSSING_FRAME_H

/** \class CrossingFrame
 *
 * CrossingFrame is the result of the Sim Mixing Module
 *
 * \author Ursula Berthon, Claude Charlot,  LLR Palaiseau
 *
 * \version   1st Version July 2005
 * \version   2nd Version Sep 2005
 *
 ************************************************************/

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/Framework/interface/Event.h"

#include <vector>
#include <string>
#include <map>
#include <utility>
using namespace edm;

//  class CrossingFrame :public BCrossingFrame
  class CrossingFrame 
    { 

    public:
      // con- and destructors

      CrossingFrame():  bunchSpace_(75), firstCrossing_(0), lastCrossing_(0) {;}
      CrossingFrame(int minb, int maxb, int bunchsp, std::vector<std::string> muonsubdetectors, std::vector<std::string> trackersubdetectors,std::vector<std::string> calosubdetectors);

      ~CrossingFrame();

     

      // methods
      void addSignalSimHits(const std::string subdet, const edm::PSimHitContainer *);
      void addSignalCaloHits(const std::string subdet, const edm::PCaloHitContainer *);
      void addSignalTracks(const edm::EmbdSimTrackContainer *);
      void addSignalVertices(const edm::EmbdSimVertexContainer *);
      void addPileupSimHits(const int bcr, const std::string subdet, const edm::PSimHitContainer *, int trackoffset, bool checkTof);
      void addPileupCaloHits(const int bcr, const std::string subdet, const edm::PCaloHitContainer *, int trackoffset=0);
      void addPileupTracks(const int bcr, const edm::EmbdSimTrackContainer *, int vertexoffset=0);
      void addPileupVertices(const int bcr, const edm::EmbdSimVertexContainer *, int trackoffset=0);      
      void print(int level=0) const ;
      void setEventID(edm::EventID id) {id_=id;}

      //getters
      edm::EventID getEventID() const {return id_;}
      std::pair<int,int> getBunchRange() const {return std::pair<int,int>(firstCrossing_,lastCrossing_);}
      int getBunchSpace() const {return bunchSpace_;}
      bool knownDetector  (const std::string subdet) const {return signalSimHits_.count(subdet) ? true : signalCaloHits_.count(subdet);}
      std::string getType(std::string subdet) {
	if (signalSimHits_.count(subdet)) return std::string("PSimHit");
        else if (signalCaloHits_.count(subdet)) return std::string("PCaloHit"); 
	else return std::string();}

      //templated getters for collections
      template <class T> void getSignal(const std::string subdet,std::vector<T> *&);
      void getSignal(const std::string subdet, std::vector<PSimHit>* &v) { v=&(signalSimHits_[subdet]);  }
      void getSignal(const std::string subdet, std::vector<PCaloHit> * &v) { v=&signalCaloHits_[subdet];}
      void getSignal(const std::string subdet, std::vector<EmbdSimTrack>* &v) { v=&signalTracks_;}
      void getSignal(const std::string subdet, std::vector<EmbdSimVertex>* &v) { v=&signalVertices_;}
      template <class T>  void getPileups(const std::string subdet,std::vector<std::vector<T> >*&);
      void getPileups(const std::string subdet, std::vector<std::vector<PSimHit> >*& v) { v=&(pileupSimHits_[subdet]);}
      void getPileups(const std::string subdet, std::vector<std::vector<PCaloHit> > * &v) { v=&pileupCaloHits_[subdet];} 
      void getPileups(const std::string subdet, std::vector<std::vector<EmbdSimTrack> > * &v) { v=&pileupTracks_;}
      void getPileups(const std::string subdet, std::vector<std::vector<EmbdSimVertex> > * &v) { v=&pileupVertices_;}

      // limits for tof to be considered for trackers
      static const int lowTrackTof; //nsec
      static const int highTrackTof;
      static const int minLowTof;
      static const int limHighLowTof;
					    
      private:
      void clear();

      edm::EventID id_;
      int bunchSpace_;  //in nsec
      int firstCrossing_;
      int lastCrossing_;

      // signal
      std::map <std::string, edm::PSimHitContainer> signalSimHits_;
      std::map <std::string, edm::PCaloHitContainer> signalCaloHits_;
      edm::EmbdSimTrackContainer signalTracks_;
      edm::EmbdSimVertexContainer signalVertices_;

      //pileup
      std::map <std::string, std::vector<edm::PSimHitContainer> > pileupSimHits_;
      std::map <std::string, std::vector<edm::PCaloHitContainer> > pileupCaloHits_;
      std::vector<edm::EmbdSimTrackContainer>  pileupTracks_;
      std::vector<edm::EmbdSimVertexContainer> pileupVertices_;

    };

#include<iosfwd>
#include<iostream>
std::ostream &operator<<(std::ostream& o, const CrossingFrame & c);

#endif 
