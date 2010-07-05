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

    int nprim_vtx = 0;
    std::vector<float> zpositions;
    std::vector<float> sumpT_lowpT;
    std::vector<float> sumpT_highpT;
    std::vector<int> ntrks_lowpT;
    std::vector<int> ntrks_highpT;
    

    int lastEvent = 0;

    for (TrackingVertexCollection::const_iterator iVtx = mergedVH->begin(); iVtx != mergedVH->end(); ++iVtx)
      {
	if(iVtx->eventId().bunchCrossing() == 0)
	{
	  if(iVtx->eventId().event()!=lastEvent) {
	    ++nprim_vtx; 
	    zpositions.push_back(iVtx->position().z());
	    sumpT_lowpT.push_back(0.);
	    sumpT_highpT.push_back(0.);
	    ntrks_lowpT.push_back(0);
	    ntrks_highpT.push_back(0);
	    std::cout << *iVtx << std::endl;
	    lastEvent=iVtx->eventId().event();
	  }
	}
      }


    float zpos;
    lastEvent = 0;

    for (TrackingParticleCollection::const_iterator iTrack = mergedPH->begin(); iTrack != mergedPH->end(); ++iTrack)
      {




        if(iTrack->eventId().bunchCrossing() == 0)
	  {
	      zpos = zpositions[iTrack->eventId().event()-1];
	      if(iTrack->matchedHit()>0) {
		if(fabs(iTrack->parentVertex()->position().z()-zpos)<0.1) {
		  std::cout << *iTrack << std::endl;
		  float Tpx = iTrack->p4().px();
		  float Tpy = iTrack->p4().py();
		  float TpT = sqrt(Tpx*Tpx + Tpy*Tpy);
		  if( TpT>pTcut_1_ ) {
		    sumpT_lowpT[iTrack->eventId().event()-1]+=TpT;
		    ++ntrks_lowpT[iTrack->eventId().event()-1];
		  }
		  if( TpT>pTcut_2_ ){
		    sumpT_highpT[iTrack->eventId().event()-1]+=TpT;
		    ++ntrks_highpT[iTrack->eventId().event()-1];
		  }
		}
	      }
	  }

      }


    std::cout << "No. of minbias vertices: " << nprim_vtx << std::endl;

    for(int iv = 0; iv<nprim_vtx; ++iv){
      std::cout << "Z position " << zpositions[iv] << std::endl;
      std::cout << "ntrks_lowpT " << ntrks_lowpT[iv] << std::endl;
      std::cout << "sumpT_lowpT " << sumpT_lowpT[iv] << std::endl;
      std::cout << "ntrks_highpT " << ntrks_highpT[iv] << std::endl;
      std::cout << "sumpT_highpT " << sumpT_highpT[iv] << std::endl;
    }

    PileupSummary_ = std::auto_ptr<PileupSummaryInfo>( new PileupSummaryInfo(  
								        nprim_vtx,
									zpositions,
									sumpT_lowpT,
									sumpT_highpT,
									ntrks_lowpT,
									ntrks_highpT)
						  );

    event.put(PileupSummary_);

}


DEFINE_FWK_MODULE(PileupInformation);
