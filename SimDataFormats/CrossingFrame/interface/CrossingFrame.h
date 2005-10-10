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
#include "FWCore/MixMod/interface/BCrossingFrame.h"

#include <vector>
#include <string>
#include <map>

  class CrossingFrame :public BCrossingFrame
    { 

    public:
      // con- and destructors

      CrossingFrame() {    printf ("default constructor CF called\n");fflush(stdout);
;}
      CrossingFrame(int minb, int maxb, int bunchsp, std::vector<std::string> trackersubdetectors,std::vector<std::string> calosubdetectors);

      ~CrossingFrame();

     

      // methods
      void addSignalSimHits(const std::string subdet, const edm::PSimHitContainer *);
      void addSignalCaloHits(const std::string subdet, const edm::PCaloHitContainer *);
      void addSignalTracks(const edm::EmbdSimTrackContainer *);
      void addSignalVertices(const edm::EmbdSimVertexContainer *);
      void addPileupSimHits(const int bcr, const std::string subdet, const edm::PSimHitContainer *);
      void addPileupCaloHits(const int bcr, const std::string subdet, const edm::PCaloHitContainer *);
      void addPileupTracks(const int bcr, const edm::EmbdSimTrackContainer *);
      void addPileupVertices(const int bcr, const edm::EmbdSimVertexContainer *);      
      void print(int level=0) const ;
 
    private:
      void clear();

 
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
 

#endif 
