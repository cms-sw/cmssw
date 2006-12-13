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
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/Common/interface/EventID.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <string>
#include <map>
#include <utility>

  class CrossingFrame 
    { 

    public:
      // con- and destructors

      CrossingFrame():  bunchSpace_(75), firstCrossing_(0), lastCrossing_(0) {;}
      CrossingFrame(int minb, int maxb, int bunchsp, std::vector<std::string> simHitSubdetectors,std::vector<std::string> caloSubdetectors);

      ~CrossingFrame();

     

      // methods
      void addSignalSimHits(const std::string subdet, const edm::PSimHitContainer *);
      void addSignalCaloHits(const std::string subdet, const edm::PCaloHitContainer *);
      void addSignalTracks(const edm::SimTrackContainer *);
      void addSignalVertices(const edm::SimVertexContainer *);
      //      void addPileupSimHits(const int bcr, const std::string subdet, const edm::PSimHitContainer *, int evtId, int trackoffset, bool checkTof);
      void addPileupSimHits(const int bcr, const std::string subdet, const edm::PSimHitContainer *, int evtId, bool checkTof);
      void addPileupCaloHits(const int bcr, const std::string subdet, const edm::PCaloHitContainer *, int evtId);
      void addPileupTracks(const int bcr, const edm::SimTrackContainer *,  int evtId, int vertexoffset);
      void addPileupVertices(const int bcr, const edm::SimVertexContainer *, int evtId);      
      void print(int level=0) const ;
      void setEventID(edm::EventID id) {id_=id;}

      //getters
      edm::EventID getEventID() const {return id_;}
      std::pair<int,int> getBunchRange() const {return std::pair<int,int>(firstCrossing_,lastCrossing_);}
      int getBunchSpace() const {return bunchSpace_;}
      //signal???      bool knownDetector  (const std::string subdet) const {return signalSimHits_.count(subdet) ? true : signalCaloHits_.count(subdet);}
      bool knownDetector  (const std::string subdet) const {return pileupSimHits_.count(subdet) ? true : pileupCaloHits_.count(subdet);}
      std::string getType(std::string subdet) {
	if (signalSimHits_.count(subdet)) return std::string("PSimHit");
        else if (signalCaloHits_.count(subdet)) return std::string("PCaloHit"); 
	else return std::string();}

      //templated getters for collections
      template <class T> void getSignal(const std::string subdet,std::vector<T> *&);
      void getSignal(const std::string subdet, std::vector<PSimHit>* &v) { v=&(signalSimHits_[subdet]);  }
      void getSignal(const std::string subdet, std::vector<PCaloHit> * &v) { v=&signalCaloHits_[subdet];}
      void getSignal(const std::string subdet, std::vector<SimTrack>* &v) { v=&signalTracks_;}
      void getSignal(const std::string subdet, std::vector<SimVertex>* &v) { v=&signalVertices_;}
      template <class T>  void getPileups(const std::string subdet,std::vector<std::vector<T> >*&);
      void getPileups(const std::string subdet, std::vector<std::vector<PSimHit> >*& v) { v=&(pileupSimHits_[subdet]);}
      void getPileups(const std::string subdet, std::vector<std::vector<PCaloHit> > * &v) { v=&pileupCaloHits_[subdet];} 
      void getPileups(const std::string subdet, std::vector<std::vector<SimTrack> > * &v) { v=&pileupTracks_;}
      void getPileups(const std::string subdet, std::vector<std::vector<SimVertex> > * &v) { v=&pileupVertices_;}
      // this does not compile....
/*       template <class T>  void getPileups(const std::string subdet,std::vector<T,std::allocator<T> >*& v,int bcr) const */
/* 	{if ( bcr<firstCrossing_ || bcr >lastCrossing_ ) { */
/* 	  edm::LogWarning("")<<" BunchCrossing nr "<<bcr<<" does not exist!"; */
/* 	  return; } */
/* 	else {std::vector<std::vector<T> > vv; */
/* 	getPileups(subdet,&vv); v=&(*vv)[bcr]; return;} */
/* 	} */
	
      // non-templated getters per bunchcrossing
/*       std::vector<PSimHit> * getPileupSimHits(const std::string subdet, const int bcr) {if ( bcr<firstCrossing_ || bcr >lastCrossing_ ) { */
/* 	edm::LogWarning("")<<" BunchCrossing nr "<<bcr<<" does not exist!"; */
/*         return 0;} */
/*       else return &pileupSimHits_[subdet][bcr-firstCrossing_];}  */

/*       std::vector<PCaloHit> *getPileupCaloHits(const std::string subdet, const int bcr) {if ( bcr<firstCrossing_ || bcr >lastCrossing_ ) { */
/* 	edm::LogWarning("")<<" BunchCrossing nr "<<bcr<<" does not exist!"; */
/*         return 0;} */
/*       else return &pileupCaloHits_[subdet][bcr-firstCrossing_];}  */

      const std::vector<SimTrack> &getPileupTracks(const int bcr) const {if ( bcr<firstCrossing_ || bcr >lastCrossing_ ) {
	edm::LogWarning("")<<" BunchCrossing nr "<<bcr<<" does not exist!";
	std::vector<SimTrack> v;
        return v;}
      else return pileupTracks_[bcr-firstCrossing_];} 

/*       std::vector<SimVertex> *getPileupVertices(const int bcr) {if ( bcr<firstCrossing_ || bcr >lastCrossing_ ) { */
/* 	edm::LogWarning("")<<" BunchCrossing nr "<<bcr<<" does not exist!"; */
/* 	return 0;} */
/*       else return &pileupVertices_[bcr-firstCrossing_];}  */

      // getters for nr of objects - mind that objects are stored in vectors from 0 on!
      unsigned int getNrSignalTracks() const { return signalTracks_.size();}
      unsigned int getNrPileupTracks(const int bcr) const {if ( bcr<firstCrossing_ || bcr >lastCrossing_ ) {
	edm::LogWarning("")<<" BunchCrossing nr "<<bcr<<" does not exist!";
	return 0;}
      else return pileupTracks_[bcr-firstCrossing_].size();}
      unsigned int getNrSignalVerticess() const { return signalVertices_.size();}
      unsigned int getNrPileupVertices(int bcr) const {if ( bcr<firstCrossing_ || bcr >lastCrossing_ ) {
	edm::LogWarning("")<<" BunchCrossing nr "<<bcr<<" does not exist!";
	return 0;}
      else return pileupVertices_[bcr-firstCrossing_].size();}
      
      unsigned int getNrSignalSimHits(const std::string subdet) const;
      unsigned int getNrSignalCaloHits(const std::string subdet) const ;
      unsigned int getNrPileupSimHits(const std::string subdet, const int bcr) const ;
      unsigned int getNrPileupCaloHits(const std::string subdet, const int bcr) const ;

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
      edm::SimTrackContainer signalTracks_;
      edm::SimVertexContainer signalVertices_;

      //pileup
      std::map <std::string, std::vector<edm::PSimHitContainer> > pileupSimHits_;
      std::map <std::string, std::vector<edm::PCaloHitContainer> > pileupCaloHits_;
      std::vector<edm::SimTrackContainer>  pileupTracks_;
      std::vector<edm::SimVertexContainer> pileupVertices_;

    };

#include<iosfwd>
#include<iostream>
std::ostream &operator<<(std::ostream& o, const CrossingFrame & c);

#endif 
