// File: PileupInformation.cc
// Description:  adds pileup information object to event
// Author:  Mike Hildreth
//
// Adds a vector of PileupSummaryInfo objects to the event. 
// One for each bunch crossing.
//
//--------------------------------------------

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "SimGeneral/PileupInformation/interface/PileupInformation.h"


PileupInformation::PileupInformation(const edm::ParameterSet & config) 
{
    // Initialize global parameters

    pTcut_1_                = 0.1;
    pTcut_2_                = 0.5; // defaults                                                       
    distanceCut_            = config.getParameter<double>("vertexDistanceCut");
    volumeRadius_           = config.getParameter<double>("volumeRadius");
    volumeZ_                = config.getParameter<double>("volumeZ");
    pTcut_1_                = config.getParameter<double>("pTcut_1");
    pTcut_2_                = config.getParameter<double>("pTcut_2");

    PileupInfoLabel_        = consumes<PileupMixingContent>(config.getParameter<edm::InputTag>("PileupMixingLabel"));

    LookAtTrackingTruth_    = config.getUntrackedParameter<bool>("doTrackTruth");

    trackingTruthT_          = mayConsume<TrackingParticleCollection>(config.getParameter<edm::InputTag>("TrackingParticlesLabel"));
    trackingTruthV_          = mayConsume<TrackingVertexCollection>(config.getParameter<edm::InputTag>("TrackingParticlesLabel"));

    MessageCategory_        = "PileupInformation";

    edm::LogInfo (MessageCategory_) << "Setting up PileupInformation";
    edm::LogInfo (MessageCategory_) << "Vertex distance cut set to " << distanceCut_  << " mm";
    edm::LogInfo (MessageCategory_) << "Volume radius set to "       << volumeRadius_ << " mm";
    edm::LogInfo (MessageCategory_) << "Volume Z      set to "       << volumeZ_      << " mm";
    edm::LogInfo (MessageCategory_) << "Lower pT Threshold set to "       << pTcut_1_      << " GeV";
    edm::LogInfo (MessageCategory_) << "Upper pT Threshold set to "       << pTcut_2_      << " GeV";


    produces< std::vector<PileupSummaryInfo> >();
    //produces<PileupSummaryInfo>();
}


void PileupInformation::produce(edm::Event &event, const edm::EventSetup & setup)
{

  std::auto_ptr<std::vector<PileupSummaryInfo> > PSIVector(new std::vector<PileupSummaryInfo>);

  edm::Handle< PileupMixingContent > MixingPileup;  // Get True pileup information from MixingModule
  event.getByToken(PileupInfoLabel_, MixingPileup);

  std::vector<int> BunchCrossings;
  std::vector<int> Interactions_Xing;
  std::vector<float> TrueInteractions_Xing;

  const PileupMixingContent* MixInfo = MixingPileup.product();

  if(MixInfo) {  // extract information - way easier than counting vertices

    const std::vector<int> bunchCrossing = MixInfo->getMix_bunchCrossing();
    const std::vector<int> interactions = MixInfo->getMix_Ninteractions();
    const std::vector<float> TrueInteractions = MixInfo->getMix_TrueInteractions();


    for(int ib=0; ib<(int)bunchCrossing.size(); ++ib){
      //      std::cout << " bcr, nint " << bunchCrossing[ib] << " " << interactions[ib] << std::endl;
      BunchCrossings.push_back(bunchCrossing[ib]);
      Interactions_Xing.push_back(interactions[ib]);
      TrueInteractions_Xing.push_back(TrueInteractions[ib]);
    }
  }
  else{ // have to throw an exception..

    throw cms::Exception("PileupInformation") << " PileupMixingContent is missing from the event.\n" 
                                                 "There must be some breakdown in the Simulation Chain.\n"
                                                 "You must run the MixingModule before calling this routine."; 

    /**  The following code is not valid when there are no Crossing Frames, since all of the vertices 
        will not be in memory at the same time

    // Collect all the simvertex from the crossing frame                                            
    edm::Handle<CrossingFrame<SimVertex> > cfSimVertexes;
    if( event.getByToken(simHitToken_, cfSimVertexes) ) {

      // Create a mix collection from one simvertex collection                                        
      simVertexes_ = std::auto_ptr<MixCollection<SimVertex> >( new MixCollection<SimVertex>(cfSimVertexes.product()) );

      int index = 0;
      // Solution to the problem of not having vertexId
      //    bool FirstL = true;
      EncodedEventIdToIndex vertexId;
      EncodedEventId oldEventId;
      int oldBX = -1000;

      std::vector<int> BunchCrossings2;
      std::list<int> Interactions_Xing2;


      // Loop for finding repeated vertexId (vertexId problem hack)                                   
      for (MixCollection<SimVertex>::MixItr iterator = simVertexes_->begin(); iterator != simVertexes_->end(); ++iterator, ++index)
	{
	  //      std::cout << " SimVtx eventid, vertexid " << iterator->eventId().event() << " " << iterator->eventId().bunchCrossing() << std::endl;
	  if (!index || iterator->eventId() != oldEventId)
	    {
	      if(iterator->eventId().bunchCrossing()==0 && iterator->eventId().event()==0){
		continue;
	      }
	      if(iterator->eventId().bunchCrossing() != oldBX) {
		BunchCrossings2.push_back(iterator->eventId().bunchCrossing());
		Interactions_Xing2.push_back(iterator->eventId().event());
		oldBX = iterator->eventId().bunchCrossing();
	      }
	      else { Interactions_Xing2.pop_back();
		Interactions_Xing2.push_back(iterator->eventId().event());
	      }


	      oldEventId = iterator->eventId();
	      continue;
	    }

	}

      std::vector<int>::iterator viter;
      std::list<int>::iterator liter = Interactions_Xing2.begin();

      for(viter = BunchCrossings2.begin(); viter != BunchCrossings2.end(); ++viter, ++liter){
	//std::cout << " bcr, nint from VTX " << (*viter) << " " << (*liter) << std::endl;
	BunchCrossings.push_back((*viter));
	Interactions_Xing.push_back((*liter));
	TrueInteractions_Xing.push_back(-1.);  // no idea what the true number is
      }

    } // end of "did we find vertices?"
   // end of look at SimVertices   **/  // end of cut
  }

  //Now, get information on valid particles that look like they could be in the tracking volume


  zpositions.clear();
  sumpT_lowpT.clear();
  sumpT_highpT.clear();
  ntrks_lowpT.clear();
  ntrks_highpT.clear();
  event_index_.clear();

  int lastEvent = 0; // zero is the true MC hard-scatter event

  // int lastBunchCrossing = 0; // 0 is the true bunch crossing, should always come first.

  bool HaveTrackingParticles = false;

  edm::Handle<TrackingParticleCollection> mergedPH;
  edm::Handle<TrackingVertexCollection>   mergedVH;

  TrackingVertexCollection::const_iterator iVtx;
  TrackingVertexCollection::const_iterator iVtxTest;
  TrackingParticleCollection::const_iterator iTrackTest;

  if( LookAtTrackingTruth_ ){

    if(event.getByToken(trackingTruthT_, mergedPH) && event.getByToken(trackingTruthV_, mergedVH)) {

      HaveTrackingParticles = true;

      iVtxTest = mergedVH->begin();
      iTrackTest = mergedPH->begin();
    }

  }

  int nminb_vtx = 0;
  //  bool First = true;
  //  bool flag_new = false;

  std::vector<int>::iterator BXIter;
  std::vector<int>::iterator InteractionsIter = Interactions_Xing.begin();
  std::vector<float>::iterator TInteractionsIter = TrueInteractions_Xing.begin();

  // loop over the bunch crossings and interactions we have extracted 

  for( BXIter = BunchCrossings.begin(); BXIter != BunchCrossings.end(); ++BXIter, ++InteractionsIter, ++TInteractionsIter) {

    //std::cout << "looking for BX: " << (*BXIter) << std::endl;

    if(HaveTrackingParticles) {  // leave open the option of not producing TrackingParticles and just keeping # interactions

      for (iVtx = iVtxTest; iVtx != mergedVH->end(); ++iVtx) {     

	if(iVtx->eventId().bunchCrossing() == (*BXIter) ) { // found first vertex in this bunch crossing

	  if(iVtx->eventId().event() != lastEvent) {

	    //std::cout << "BX,event " << iVtx->eventId().bunchCrossing() << " " << iVtx->eventId().event() << std::endl;

	    float zpos = 0.;
	    zpos = iVtx->position().z();
	    zpositions.push_back(zpos);  //save z position of each vertex
	    sumpT_lowpT.push_back(0.);
	    sumpT_highpT.push_back(0.);
	    ntrks_lowpT.push_back(0);
	    ntrks_highpT.push_back(0);

	    lastEvent = iVtx->eventId().event();
	    iVtxTest = --iVtx; // just for security

	    // turns out events aren't sequential... save map of indices

	    event_index_.insert(myindex::value_type(lastEvent,nminb_vtx));
	     
	    ++nminb_vtx;

	    continue;
	  }
	}
      }
    
      // next loop over tracks to get information

      for (TrackingParticleCollection::const_iterator iTrack = iTrackTest; iTrack != mergedPH->end(); ++iTrack)
	{
	  bool FoundTrk = false;

	  float zpos=0.;

	  if(iTrack->eventId().bunchCrossing() == (*BXIter) && iTrack->eventId().event() > 0 )
	    {
	      FoundTrk = true;
	      int correct_index = event_index_[iTrack->eventId().event()];

	      //std::cout << " track index, correct index " << iTrack->eventId().event() << " " << correct_index << std::endl;

	      zpos = zpositions[correct_index];
	      if(iTrack->matchedHit()>0) {
		if(fabs(iTrack->parentVertex()->position().z()-zpos)<0.1) {  //make sure track really comes from this vertex
		  //std::cout << *iTrack << std::endl;                                              
		  float Tpx = iTrack->p4().px();
		  float Tpy = iTrack->p4().py();
		  float TpT = sqrt(Tpx*Tpx + Tpy*Tpy);
		  if( TpT>pTcut_1_ ) {
		    sumpT_lowpT[correct_index]+=TpT;
		    ++ntrks_lowpT[correct_index];
		  }
		  if( TpT>pTcut_2_ ){
		    sumpT_highpT[correct_index]+=TpT;
		    ++ntrks_highpT[correct_index];
		  }
		}
	      }
	    }
	  else{
	    if(FoundTrk) {

	      iTrackTest = --iTrack;  // reset so we can start over next time
	      --iTrackTest;  // just to be sure
	      break;
	    }
	
	  }
	
	} // end of track loop

    } // end of check that we have TrackingParticles to begin with...


    // now that we have all of the track information for a given bunch crossing, 
    // make PileupSummary for this one and move on

    //	  std::cout << "Making PSI for bunch " << lastBunchCrossing << std::endl;

    if(!HaveTrackingParticles) { // stick in one value so we don't have empty vectors

      zpositions.push_back(-999.);  
      sumpT_lowpT.push_back(0.);
      sumpT_highpT.push_back(0.);
      ntrks_lowpT.push_back(0);
      ntrks_highpT.push_back(0);

    }

    PileupSummaryInfo	PSI_bunch = PileupSummaryInfo(
						      (*InteractionsIter),
						      zpositions,
						      sumpT_lowpT,
						      sumpT_highpT,
						      ntrks_lowpT,
						      ntrks_highpT,
						      (*BXIter),
						      (*TInteractionsIter)
						      );

    //std::cout << " " << std::endl;
    //std::cout << "Adding Bunch Crossing, nint " << (*BXIter) << " " <<  (*InteractionsIter) << std::endl;
 
    //for(int iv = 0; iv<(*InteractionsIter); ++iv){
    	    
    // std::cout << "Z position " << zpositions[iv] << std::endl;
    // std::cout << "ntrks_lowpT " << ntrks_lowpT[iv] << std::endl;
    // std::cout << "sumpT_lowpT " << sumpT_lowpT[iv] << std::endl;
    // std::cout << "ntrks_highpT " << ntrks_highpT[iv] << std::endl;
    // std::cout << "sumpT_highpT " << sumpT_highpT[iv] << std::endl;
    //}

    PSIVector->push_back(PSI_bunch);

    // if(HaveTrackingParticles) lastBunchCrossing = iVtx->eventId().bunchCrossing();

    event_index_.clear();
    zpositions.clear();
    sumpT_lowpT.clear();
    sumpT_highpT.clear();
    ntrks_lowpT.clear();
    ntrks_highpT.clear();
    nminb_vtx = 0;
    lastEvent=0;


  } // end of loop over bunch crossings

  // put our vector of PileupSummaryInfo objects into the event.

  event.put(PSIVector);


}


DEFINE_FWK_MODULE(PileupInformation);
