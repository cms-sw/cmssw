// File: PileupInformation.cc
// Description:  adds pileup information object to event
// Author:  Mike Hildreth
//
//--------------------------------------------

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

    trackingTruth_  = config.getParameter<std::string>("TrackingParticlesLabel");

    MessageCategory_       = "PileupInformation";

    edm::LogInfo (MessageCategory_) << "Setting up PileupInformation";
    edm::LogInfo (MessageCategory_) << "Vertex distance cut set to " << distanceCut_  << " mm";
    edm::LogInfo (MessageCategory_) << "Volume radius set to "       << volumeRadius_ << " mm";
    edm::LogInfo (MessageCategory_) << "Volume Z      set to "       << volumeZ_      << " mm";
    edm::LogInfo (MessageCategory_) << "Lower pT Threshold set to "       << pTcut_1_      << " GeV";
    edm::LogInfo (MessageCategory_) << "Upper pT Threshold set to "       << pTcut_2_      << " GeV";

    produces<PileupSummaryInfo>();
}


void PileupInformation::produce(edm::Event &event, const edm::EventSetup & setup)
{

    edm::Handle<TrackingParticleCollection> mergedPH;
    edm::Handle<TrackingVertexCollection>   mergedVH;

    event.getByLabel(trackingTruth_, mergedPH);
    event.getByLabel(trackingTruth_, mergedVH);

    int nminb_vtx = 0;

    zpositions.clear();
    sumpT_lowpT.clear();
    sumpT_highpT.clear();
    ntrks_lowpT.clear();
    ntrks_highpT.clear();
    event_index_.clear();

    int lastEvent = 0;


    for (TrackingVertexCollection::const_iterator iVtx = mergedVH->begin(); iVtx != mergedVH->end(); ++iVtx)
      {
        if(iVtx->eventId().bunchCrossing() == 0)
	  {
	    if(iVtx->eventId().event()!=lastEvent) {
	      float zpos = 0.;
	      zpos = iVtx->position().z();
	      zpositions.push_back(zpos);
	      sumpT_lowpT.push_back(0.);
	      sumpT_highpT.push_back(0.);
	      ntrks_lowpT.push_back(0);
	      ntrks_highpT.push_back(0);
	      //      std::cout << *iVtx << std::endl;                                                
	      lastEvent=iVtx->eventId().event();

	      // turns out events aren't sequential... save map of indices

	      event_index_.insert(myindex::value_type(lastEvent,nminb_vtx));
	      ++nminb_vtx;
	    }
	  }
      }

    float zpos = 0.;

    for (TrackingParticleCollection::const_iterator iTrack = mergedPH->begin(); iTrack != mergedPH->end(); ++iTrack)
      {

        if(iTrack->eventId().bunchCrossing() == 0 && iTrack->eventId().event() > 0 )
          {
            int correct_index = event_index_[iTrack->eventId().event()];
	    zpos = zpositions[correct_index];
	    if(iTrack->matchedHit()>0) {
	      if(fabs(iTrack->parentVertex()->position().z()-zpos)<0.1) {
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

      }




    //    for(int iv = 0; iv<nminb_vtx; ++iv){
    //  std::cout << "Z position " << zpositions[iv] << std::endl;
    //  std::cout << "ntrks_lowpT " << ntrks_lowpT[iv] << std::endl;
    //  std::cout << "sumpT_lowpT " << sumpT_lowpT[iv] << std::endl;
    //  std::cout << "ntrks_highpT " << ntrks_highpT[iv] << std::endl;
    //  std::cout << "sumpT_highpT " << sumpT_highpT[iv] << std::endl;
    //}

    std::auto_ptr<PileupSummaryInfo> PileupSummary_;

    PileupSummary_ = std::auto_ptr<PileupSummaryInfo>( new PileupSummaryInfo(
									     nminb_vtx,
									     zpositions,
									     sumpT_lowpT,
									     sumpT_highpT,
									     ntrks_lowpT,
									     ntrks_highpT)
						       );

    event.put(PileupSummary_);

    zpositions.clear();
    sumpT_lowpT.clear();
    sumpT_highpT.clear();
    ntrks_lowpT.clear();
    ntrks_highpT.clear();



}


DEFINE_FWK_MODULE(PileupInformation);
