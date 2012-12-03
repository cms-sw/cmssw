#include "Validation/RecoTrack/interface/MultiTrackValidatorGenPs.h"
#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/QuickTrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"
#include "SimTracker/TrackAssociation/plugins/CosmicParametersDefinerForTPESProducer.h"
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoFactory.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Ref.h"

#include "TMath.h"
#include <TF1.h>

//#include <iostream>

using namespace std;
using namespace edm;

MultiTrackValidatorGenPs::MultiTrackValidatorGenPs(const edm::ParameterSet& pset):MultiTrackValidator(pset){

  gpSelector = GenParticleSelector(pset.getParameter<double>("ptMinGP"),
				   pset.getParameter<double>("minRapidityGP"),
				   pset.getParameter<double>("maxRapidityGP"),
				   pset.getParameter<double>("tipGP"),
				   pset.getParameter<double>("lipGP"),
				   pset.getParameter<bool>("chargedOnlyGP"),
				   pset.getParameter<int>("statusGP"),
				   pset.getParameter<std::vector<int> >("pdgIdGP"));

}

MultiTrackValidatorGenPs::~MultiTrackValidatorGenPs(){}


void MultiTrackValidatorGenPs::analyze(const edm::Event& event, const edm::EventSetup& setup){
  using namespace reco;
  
  edm::LogInfo("TrackValidator") << "\n====================================================" << "\n"
				 << "Analyzing new event" << "\n"
				 << "====================================================\n" << "\n";

  edm::ESHandle<ParametersDefinerForTP> parametersDefinerTP; 
  setup.get<TrackAssociatorRecord>().get(parametersDefiner,parametersDefinerTP);    
  
  edm::Handle<GenParticleCollection>  TPCollectionHeff ;
  event.getByLabel(label_tp_effic,TPCollectionHeff);
  const GenParticleCollection tPCeff = *(TPCollectionHeff.product());
  
  edm::Handle<GenParticleCollection>  TPCollectionHfake ;
  event.getByLabel(label_tp_fake,TPCollectionHfake);
  const GenParticleCollection tPCfake = *(TPCollectionHfake.product());
  
  //if (tPCeff.size()==0) {edm::LogInfo("TrackValidator") 
  //<< "TP Collection for efficiency studies has size = 0! Skipping Event." ; return;}
  //if (tPCfake.size()==0) {edm::LogInfo("TrackValidator") 
  //<< "TP Collection for fake rate studies has size = 0! Skipping Event." ; return;}
  
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  event.getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      
  
  edm::Handle< vector<PileupSummaryInfo> > puinfoH;
  event.getByLabel(label_pileupinfo,puinfoH);
  PileupSummaryInfo puinfo;      
  
  for (unsigned int puinfo_ite=0;puinfo_ite<(*puinfoH).size();++puinfo_ite){ 
    if ((*puinfoH)[puinfo_ite].getBunchCrossing()==0){
      puinfo=(*puinfoH)[puinfo_ite];
      break;
    }
  }
  
  int w=0; //counter counting the number of sets of histograms
  for (unsigned int ww=0;ww<associators.size();ww++){

    associatorByChi2 = dynamic_cast<const TrackAssociatorByChi2*>(associator[ww]);
    if (associatorByChi2==0) continue;

    for (unsigned int www=0;www<label.size();www++){
      //
      //get collections from the event
      //
      edm::Handle<View<Track> >  trackCollection;
      if(!event.getByLabel(label[www], trackCollection)&&ignoremissingtkcollection_)continue;
      //if (trackCollection->size()==0) 
      //edm::LogInfo("TrackValidator") << "TrackCollection size = 0!" ; 
      //continue;
      //}
      reco::RecoToGenCollection recGenColl;
      reco::GenToRecoCollection genRecColl;
      //associate tracks
      if(UseAssociators){
	edm::LogVerbatim("TrackValidator") << "Analyzing " 
					   << label[www].process()<<":"
					   << label[www].label()<<":"
					   << label[www].instance()<<" with "
					   << associators[ww].c_str() <<"\n";
	
	LogTrace("TrackValidator") << "Calling associateRecoToGen method" << "\n";
	recGenColl=associatorByChi2->associateRecoToGen(trackCollection,
							TPCollectionHfake,
							&event);
	LogTrace("TrackValidator") << "Calling associateGenToReco method" << "\n";
	genRecColl=associatorByChi2->associateGenToReco(trackCollection,
							TPCollectionHeff, 
							&event);
      }
      else{
	edm::LogVerbatim("TrackValidator") << "Analyzing " 
					   << label[www].process()<<":"
					   << label[www].label()<<":"
					   << label[www].instance()<<" with "
					   << associatormap.process()<<":"
					   << associatormap.label()<<":"
					   << associatormap.instance()<<"\n";
	
	Handle<reco::GenToRecoCollection > gentorecoCollectionH;
	event.getByLabel(associatormap,gentorecoCollectionH);
	genRecColl= *(gentorecoCollectionH.product()); 
	
	Handle<reco::RecoToGenCollection > recotogenCollectionH;
	event.getByLabel(associatormap,recotogenCollectionH);
	recGenColl= *(recotogenCollectionH.product()); 
      }



      // ########################################################
      // fill simulation histograms (LOOP OVER TRACKINGPARTICLES)
      // ########################################################

      //compute number of tracks per eta interval
      //
      edm::LogVerbatim("TrackValidator") << "\n# of GenParticles: " << tPCeff.size() << "\n";
      int ats(0);  	  //This counter counts the number of simTracks that are "associated" to recoTracks
      int st(0);    	  //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) )
      unsigned sts(0);   //This counter counts the number of simTracks surviving the bunchcrossing cut 
      unsigned asts(0);  //This counter counts the number of simTracks that are "associated" to recoTracks surviving the bunchcrossing cut
      for (GenParticleCollection::size_type i=0; i<tPCeff.size(); i++){ //loop over TPs collection for tracking efficiency
	GenParticleRef tpr(TPCollectionHeff, i);
	GenParticle* tp=const_cast<GenParticle*>(tpr.get());
	ParticleBase::Vector momentumTP; 
	ParticleBase::Point vertexTP;
	double dxyGen(0);
	double dzGen(0);
	

	//---------- THIS PART HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------
	//If the GenParticle is collison like, get the momentum and vertex at production state
	if(parametersDefiner=="LhcParametersDefinerForTP")
	  {
	    //fixme this one shold be implemented
	    if(! gpSelector(*tp)) continue;
	    momentumTP = tp->momentum();
	    vertexTP = tp->vertex();
	    //Calcualte the impact parameters w.r.t. PCA
	    ParticleBase::Vector momentum = parametersDefinerTP->momentum(event,setup,*tp);
	    ParticleBase::Point vertex = parametersDefinerTP->vertex(event,setup,*tp);
	    dxyGen = (-vertex.x()*sin(momentum.phi())+vertex.y()*cos(momentum.phi()));
	    dzGen = vertex.z() - (vertex.x()*momentum.x()+vertex.y()*momentum.y())/sqrt(momentum.perp2()) 
	      * momentum.z()/sqrt(momentum.perp2());
	  }
	//If the GenParticle is comics, get the momentum and vertex at PCA
	if(parametersDefiner=="CosmicParametersDefinerForTP")
	  {
	    //if(! cosmictpSelector(*tp,&bs,event,setup)) continue;	
	    momentumTP = parametersDefinerTP->momentum(event,setup,*tp);
	    vertexTP = parametersDefinerTP->vertex(event,setup,*tp);
	    dxyGen = (-vertexTP.x()*sin(momentumTP.phi())+vertexTP.y()*cos(momentumTP.phi()));
	    dzGen = vertexTP.z() - (vertexTP.x()*momentumTP.x()+vertexTP.y()*momentumTP.y())/sqrt(momentumTP.perp2()) 
	      * momentumTP.z()/sqrt(momentumTP.perp2());
	  }
	//---------- THE PART ABOVE HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------

	st++;   //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) )

	// in the coming lines, histos are filled using as input
	// - momentumTP 
	// - vertexTP 
	// - dxyGen
	// - dzGen

	histoProducerAlgo_->fill_generic_simTrack_histos(w,momentumTP,vertexTP, tp->collisionId());//fixme: check meaning of collisionId


	// ##############################################
	// fill RecoAssociated GenTracks' histograms
	// ##############################################
	// bool isRecoMatched(false); // UNUSED
	const reco::Track* matchedTrackPointer=0;
	std::vector<std::pair<RefToBase<Track>, double> > rt;
	if(genRecColl.find(tpr) != genRecColl.end()){
	  rt = (std::vector<std::pair<RefToBase<Track>, double> >) genRecColl[tpr];
	  if (rt.size()!=0) {
	    ats++; //This counter counts the number of simTracks that have a recoTrack associated
	    // isRecoMatched = true; // UNUSED
	    matchedTrackPointer = rt.begin()->first.get();
	    edm::LogVerbatim("TrackValidator") << "GenParticle #" << st 
					       << " with pt=" << sqrt(momentumTP.perp2()) 
					       << " associated with quality:" << rt.begin()->second <<"\n";
	  }
	}else{
	  edm::LogVerbatim("TrackValidator") 
	    << "GenParticle #" << st
	    << " with pt,eta,phi: " 
	    << sqrt(momentumTP.perp2()) << " , "
	    << momentumTP.eta() << " , "
	    << momentumTP.phi() << " , "
	    << " NOT associated to any reco::Track" << "\n";
	}
	

	
	/*
	std::vector<PSimHit> simhits=tp->trackPSimHit(DetId::Tracker);
        int nSimHits = simhits.end()-simhits.begin();
	*/
        int nSimHits = 0;

        double vtx_z_PU = vertexTP.z();
	/*
        for (size_t j = 0; j < tv.size(); j++) {
            if (tp->eventId().event() == tv[j].eventId().event()) {
                vtx_z_PU = tv[j].position().z();
                break;
            }
        }
	*/

	histoProducerAlgo_->fill_recoAssociated_simTrack_histos(w,*tp,momentumTP,vertexTP,dxyGen,dzGen,nSimHits,matchedTrackPointer,puinfo.getPU_NumInteractions(), vtx_z_PU);

	sts++;
	if (matchedTrackPointer) asts++;




      } // End  for (GenParticleCollection::size_type i=0; i<tPCeff.size(); i++){

      //if (st!=0) h_tracksSIM[w]->Fill(st);  // TO BE FIXED
      
      
      // ##############################################
      // fill recoTracks histograms (LOOP OVER TRACKS)
      // ##############################################
      edm::LogVerbatim("TrackValidator") << "\n# of reco::Tracks with "
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()
					 << ": " << trackCollection->size() << "\n";

      //int sat(0); //This counter counts the number of recoTracks that are associated to GenTracks from Signal only
      int at(0); //This counter counts the number of recoTracks that are associated to GenTracks
      int rT(0); //This counter counts the number of recoTracks in general


      // dE/dx
      // at some point this could be generalized, with a vector of tags and a corresponding vector of Handles
      // I'm writing the interface such to take vectors of ValueMaps
      edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx1Handle;
      edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx2Handle;
      std::vector<edm::ValueMap<reco::DeDxData> > v_dEdx;
      v_dEdx.clear();
      //std::cout << "PIPPO: label is " << label[www] << std::endl;
      if (label[www].label()=="generalTracks") {
	try {
	  event.getByLabel(m_dEdx1Tag, dEdx1Handle);
	  const edm::ValueMap<reco::DeDxData> dEdx1 = *dEdx1Handle.product();
	  event.getByLabel(m_dEdx2Tag, dEdx2Handle);
	  const edm::ValueMap<reco::DeDxData> dEdx2 = *dEdx2Handle.product();
	  v_dEdx.push_back(dEdx1);
	  v_dEdx.push_back(dEdx2);
	} catch (cms::Exception e){
	  LogTrace("TrackValidator") << "exception found: " << e.what() << "\n";
	}
      }
      //end dE/dx

      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){

	RefToBase<Track> track(trackCollection, i);
	rT++;
	
	bool isSigGenMatched(false);
	bool isGenMatched(false);
	bool isChargeMatched(true);
	int numAssocRecoTracks = 0;
        int tpbx = 0;
	int nSimHits = 0;
	double sharedFraction = 0.;
	std::vector<std::pair<GenParticleRef, double> > tp;
	if(recGenColl.find(track) != recGenColl.end()){
	  tp = recGenColl[track];
	  if (tp.size()!=0) {
	    /*
	    std::vector<PSimHit> simhits=tp[0].first->trackPSimHit(DetId::Tracker);
            nSimHits = simhits.end()-simhits.begin();
	    */
            sharedFraction = tp[0].second;
	    isGenMatched = true;
	    if (tp[0].first->charge() != track->charge()) isChargeMatched = false;
	    if(genRecColl.find(tp[0].first) != genRecColl.end()) numAssocRecoTracks = genRecColl[tp[0].first].size();
	    //std::cout << numAssocRecoTracks << std::endl;
	    /*
	    tpbx = tp[0].first->eventId().bunchCrossing();
	    */
	    at++;
	    for (unsigned int tp_ite=0;tp_ite<tp.size();++tp_ite){ 
              GenParticle trackpart = *(tp[tp_ite].first);
	      /*
	      if ((trackpart.eventId().event() == 0) && (trackpart.eventId().bunchCrossing() == 0)){
	      	isSigGenMatched = true;
		sat++;
		break;
	      }
	      */
            }
	    edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt() 
					       << " associated with quality:" << tp.begin()->second <<"\n";
	  }
	} else {
	  edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
					     << " NOT associated to any GenParticle" << "\n";		  
	}
	

	histoProducerAlgo_->fill_generic_recoTrack_histos(w,*track,bs.position(),isGenMatched,isSigGenMatched, isChargeMatched, numAssocRecoTracks, puinfo.getPU_NumInteractions(), tpbx, nSimHits, sharedFraction);

	// dE/dx
	//	reco::TrackRef track2  = reco::TrackRef( trackCollection, i );
	if (v_dEdx.size() > 0) histoProducerAlgo_->fill_dedx_recoTrack_histos(w,track, v_dEdx);
	//if (v_dEdx.size() > 0) histoProducerAlgo_->fill_dedx_recoTrack_histos(track2, v_dEdx);


	//Fill other histos
 	//try{ //Is this really necessary ????

	if (tp.size()==0) continue;	

	histoProducerAlgo_->fill_simAssociated_recoTrack_histos(w,*track);

	GenParticleRef tpr = tp.begin()->first;
	
	/* TO BE FIXED LATER
	if (associators[ww]=="TrackAssociatorByChi2"){
	  //association chi2
	  double assocChi2 = -tp.begin()->second;//in association map is stored -chi2
	  h_assochi2[www]->Fill(assocChi2);
	  h_assochi2_prob[www]->Fill(TMath::Prob((assocChi2)*5,5));
	}
	else if (associators[ww]=="quickTrackAssociatorByHits"){
	  double fraction = tp.begin()->second;
	  h_assocFraction[www]->Fill(fraction);
	  h_assocSharedHit[www]->Fill(fraction*track->numberOfValidHits());
	}
	*/

	  
	//Get tracking particle parameters at point of closest approach to the beamline
	ParticleBase::Vector momentumTP = parametersDefinerTP->momentum(event,setup,*(tpr.get()));
	ParticleBase::Point vertexTP = parametersDefinerTP->vertex(event,setup,*(tpr.get()));		 	 
	int chargeTP = tpr->charge();

	histoProducerAlgo_->fill_ResoAndPull_recoTrack_histos(w,momentumTP,vertexTP,chargeTP,
							     *track,bs.position());
	
	
	//TO BE FIXED
	//std::vector<PSimHit> simhits=tpr.get()->trackPSimHit(DetId::Tracker);
	//nrecHit_vs_nsimHit_rec2sim[w]->Fill(track->numberOfValidHits(), (int)(simhits.end()-simhits.begin() ));
	
	/*
	  } // End of try{
	  catch (cms::Exception e){
	  LogTrace("TrackValidator") << "exception found: " << e.what() << "\n";
	  }
	*/
	
      } // End of for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){

      histoProducerAlgo_->fill_trackBased_histos(w,at,rT,st);

      edm::LogVerbatim("TrackValidator") << "Total Simulated: " << st << "\n"
					 << "Total Associated (genToReco): " << ats << "\n"
					 << "Total Reconstructed: " << rT << "\n"
					 << "Total Associated (recoToGen): " << at << "\n"
					 << "Total Fakes: " << rT-at << "\n";

      w++;
    } // End of  for (unsigned int www=0;www<label.size();www++){
  } //END of for (unsigned int ww=0;ww<associators.size();ww++){

}
