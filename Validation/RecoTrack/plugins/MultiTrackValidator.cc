#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
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
#include<type_traits>


#include "TMath.h"
#include <TF1.h>
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
//#include <iostream>

using namespace std;
using namespace edm;

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;

MultiTrackValidator::MultiTrackValidator(const edm::ParameterSet& pset):MultiTrackValidatorBase(pset,consumesCollector()){
  //theExtractor = IsoDepositExtractorFactory::get()->create( extractorName, extractorPSet, consumesCollector());

  ParameterSet psetForHistoProducerAlgo = pset.getParameter<ParameterSet>("histoProducerAlgoBlock");
  string histoProducerAlgoName = psetForHistoProducerAlgo.getParameter<string>("ComponentName");
  histoProducerAlgo_ = MTVHistoProducerAlgoFactory::get()->create(histoProducerAlgoName ,psetForHistoProducerAlgo, consumesCollector());

  dirName_ = pset.getParameter<std::string>("dirName");
  assMapInput = pset.getParameter< edm::InputTag >("associatormap");
  associatormapStR = mayConsume<reco::SimToRecoCollection>(assMapInput);
  associatormapRtS = mayConsume<reco::RecoToSimCollection>(assMapInput);
  UseAssociators = pset.getParameter< bool >("UseAssociators");

  m_dEdx1Tag = mayConsume<edm::ValueMap<reco::DeDxData> >(pset.getParameter< edm::InputTag >("dEdx1Tag"));
  m_dEdx2Tag = mayConsume<edm::ValueMap<reco::DeDxData> >(pset.getParameter< edm::InputTag >("dEdx2Tag"));

  tpSelector = TrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
					pset.getParameter<double>("minRapidityTP"),
					pset.getParameter<double>("maxRapidityTP"),
					pset.getParameter<double>("tipTP"),
					pset.getParameter<double>("lipTP"),
					pset.getParameter<int>("minHitTP"),
					pset.getParameter<bool>("signalOnlyTP"),
					pset.getParameter<bool>("chargedOnlyTP"),
					pset.getParameter<bool>("stableOnlyTP"),
					pset.getParameter<std::vector<int> >("pdgIdTP"));

  cosmictpSelector = CosmicTrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
						    pset.getParameter<double>("minRapidityTP"),
						    pset.getParameter<double>("maxRapidityTP"),
						    pset.getParameter<double>("tipTP"),
						    pset.getParameter<double>("lipTP"),
						    pset.getParameter<int>("minHitTP"),
						    pset.getParameter<bool>("chargedOnlyTP"),
						    pset.getParameter<std::vector<int> >("pdgIdTP"));


  ParameterSet psetVsEta = psetForHistoProducerAlgo.getParameter<ParameterSet>("TpSelectorForEfficiencyVsEta");
  dRtpSelector = TrackingParticleSelector(psetVsEta.getParameter<double>("ptMin"),
					  psetVsEta.getParameter<double>("minRapidity"),
					  psetVsEta.getParameter<double>("maxRapidity"),
					  psetVsEta.getParameter<double>("tip"),
					  psetVsEta.getParameter<double>("lip"),
					  psetVsEta.getParameter<int>("minHit"),
					  psetVsEta.getParameter<bool>("signalOnly"),
					  psetVsEta.getParameter<bool>("chargedOnly"),
					  psetVsEta.getParameter<bool>("stableOnly"),
					  psetVsEta.getParameter<std::vector<int> >("pdgId"));

  useGsf = pset.getParameter<bool>("useGsf");
  runStandalone = pset.getParameter<bool>("runStandalone");

  _simHitTpMapTag = mayConsume<SimHitTPAssociationProducer::SimHitTPAssociationList>(pset.getParameter<edm::InputTag>("simHitTpMapTag"));

  labelTokenForDrCalculation = consumes<edm::View<reco::Track> >(pset.getParameter<edm::InputTag>("trackCollectionForDrCalculation"));

  if (!UseAssociators) {
    associators.clear();
    associators.push_back(assMapInput.label());
  } else {   
    for (auto const& associatorName : associators) {
      consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag(associatorName));
    }
  }

}


MultiTrackValidator::~MultiTrackValidator(){delete histoProducerAlgo_;}


void MultiTrackValidator::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const& setup) {

  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      ibook.cd();
      InputTag algo = label[www];
      string dirName=dirName_;
      if (algo.process()!="")
    dirName+=algo.process()+"_";
      if(algo.label()!="")
    dirName+=algo.label()+"_";
      if(algo.instance()!="")
    dirName+=algo.instance()+"_";
      if (dirName.find("Tracks")<dirName.length()){
    dirName.replace(dirName.find("Tracks"),6,"");
      }
      string assoc= associators[ww];
      if (assoc.find("Track")<assoc.length()){
    assoc.replace(assoc.find("Track"),5,"");
      }
      dirName+=assoc;
      std::replace(dirName.begin(), dirName.end(), ':', '_');

      ibook.setCurrentFolder(dirName.c_str());

      // vector of vector initialization
      histoProducerAlgo_->initialize(); //TO BE FIXED. I'D LIKE TO AVOID THIS CALL

      string subDirName = dirName + "/simulation";
      ibook.setCurrentFolder(subDirName.c_str());

      //Booking histograms concerning with simulated tracks
      histoProducerAlgo_->bookSimHistos(ibook);

      ibook.cd();
      ibook.setCurrentFolder(dirName.c_str());

      //Booking histograms concerning with reconstructed tracks
      histoProducerAlgo_->bookRecoHistos(ibook);
      if (runStandalone) histoProducerAlgo_->bookRecoHistosForStandaloneRunning(ibook);

    }//end loop www
  }// end loop ww
}


void MultiTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){
  using namespace reco;

  edm::LogInfo("TrackValidator") << "\n====================================================" << "\n"
				 << "Analyzing new event" << "\n"
				 << "====================================================\n" << "\n";

  std::vector<const reco::TrackToTrackingParticleAssociator*> associator;
  if (UseAssociators) {
    edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
    for (auto const& associatorName : associators) {
      event.getByLabel(associatorName,theAssociator);
      associator.push_back( theAssociator.product() );
    }
  }


  edm::ESHandle<ParametersDefinerForTP> parametersDefinerTPHandle;
  setup.get<TrackAssociatorRecord>().get(parametersDefiner,parametersDefinerTPHandle);
  //Since we modify the object, we must clone it
  auto parametersDefinerTP = parametersDefinerTPHandle->clone();

  edm::Handle<TrackingParticleCollection>  TPCollectionHeff ;
  event.getByToken(label_tp_effic,TPCollectionHeff);
  TrackingParticleCollection const & tPCeff = *(TPCollectionHeff.product());

  edm::Handle<TrackingParticleCollection>  TPCollectionHfake ;
  event.getByToken(label_tp_fake,TPCollectionHfake);


  if(parametersDefiner=="CosmicParametersDefinerForTP") {
    edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
    //warning: make sure the TP collection used in the map is the same used in the MTV!
    event.getByToken(_simHitTpMapTag,simHitsTPAssoc);
    parametersDefinerTP->initEvent(simHitsTPAssoc);
    cosmictpSelector.initEvent(simHitsTPAssoc);
  }


  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  event.getByToken(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot const & bs = *recoBeamSpotHandle;

  edm::Handle< std::vector<PileupSummaryInfo> > puinfoH;
  event.getByToken(label_pileupinfo,puinfoH);
  PileupSummaryInfo puinfo;

  for (unsigned int puinfo_ite=0;puinfo_ite<(*puinfoH).size();++puinfo_ite){
    if ((*puinfoH)[puinfo_ite].getBunchCrossing()==0){
      puinfo=(*puinfoH)[puinfo_ite];
      break;
    }
  }

  /*
  edm::Handle<TrackingVertexCollection> tvH;
  event.getByToken(label_tv,tvH);
  TrackingVertexCollection const & tv = *tvH;
  */

  //calculate dR for TPs
  float dR_tPCeff[(*TPCollectionHeff).size()];
  {
    int j=0;
    float etaL[(*TPCollectionHeff).size()], phiL[(*TPCollectionHeff).size()];
    bool okL[(*TPCollectionHeff).size()];
    for (   auto const & tp2 : *TPCollectionHeff) {
      okL[j]=false;
      if(tpSelector(tp2)) { //calculare dR wrt inclusive collection (also with PU, low pT, displaced)
        okL[j]=true;
        auto  && p = tp2.momentum();
        etaL[j] = etaFromXYZ(p.x(),p.y(),p.z());
        phiL[j] = atan2f(p.y(),p.x());

      } 
      ++j;
    }
    auto i=0U;
    for ( auto const & tp : *TPCollectionHeff) {
      double dR = std::numeric_limits<double>::max();
      if(dRtpSelector(tp)) {//only for those needed for efficiency!
        auto  && p = tp.momentum();
        float eta = etaFromXYZ(p.x(),p.y(),p.z());
        float phi = atan2f(p.y(),p.x());
        for (auto j=0U; j< (*TPCollectionHeff).size(); ++j ) {
	  if (i==j) {continue;}
	  if(okL[j]) { //calculare dR wrt inclusive collection (also with PU, low pT, displaced)
            auto dR_tmp = reco::deltaR2(eta, phi, etaL[j], phiL[j]);
            if (dR_tmp<dR) dR=dR_tmp;
          }
        }  // ttp2 (j)
      }
      dR_tPCeff[i++] = std::sqrt(dR);
    }  // tp
  }

  edm::Handle<View<Track> >  trackCollectionForDrCalculation;
  event.getByToken(labelTokenForDrCalculation, trackCollectionForDrCalculation);

  int w=0; //counter counting the number of sets of histograms
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      //
      //get collections from the event
      //
      edm::Handle<View<Track> >  trackCollection;
      if(!event.getByToken(labelToken[www], trackCollection)&&ignoremissingtkcollection_)continue;

      reco::RecoToSimCollection const * recSimCollP=nullptr;
      reco::SimToRecoCollection const * simRecCollP=nullptr;
      reco::RecoToSimCollection recSimCollL;
      reco::SimToRecoCollection simRecCollL;

      //associate tracks
      if(UseAssociators){
	edm::LogVerbatim("TrackValidator") << "Analyzing "
					   << label[www].process()<<":"
					   << label[www].label()<<":"
					   << label[www].instance()<<" with "
					   << associators[ww].c_str() <<"\n";

	LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
	recSimCollL = std::move(associator[ww]->associateRecoToSim(trackCollection,
                                                                   TPCollectionHfake));
         recSimCollP = &recSimCollL;
	LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
	simRecCollL = std::move(associator[ww]->associateSimToReco(trackCollection,
                                                                   TPCollectionHeff));
        simRecCollP = &simRecCollL;
      }
      else{
	edm::LogVerbatim("TrackValidator") << "Analyzing "
					   << label[www].process()<<":"
					   << label[www].label()<<":"
					   << label[www].instance()<<" with "
					   << assMapInput.process()<<":"
					   << assMapInput.label()<<":"
					   << assMapInput.instance()<<"\n";

	Handle<reco::SimToRecoCollection > simtorecoCollectionH;
	event.getByToken(associatormapStR,simtorecoCollectionH);
	simRecCollP = simtorecoCollectionH.product();

	Handle<reco::RecoToSimCollection > recotosimCollectionH;
	event.getByToken(associatormapRtS,recotosimCollectionH);
	recSimCollP = recotosimCollectionH.product();
      }

      reco::RecoToSimCollection const & recSimColl = *recSimCollP;
      reco::SimToRecoCollection const & simRecColl = *simRecCollP;
 


      // ########################################################
      // fill simulation histograms (LOOP OVER TRACKINGPARTICLES)
      // ########################################################

      //compute number of tracks per eta interval
      //
      edm::LogVerbatim("TrackValidator") << "\n# of TrackingParticles: " << tPCeff.size() << "\n";
      int ats(0);  	  //This counter counts the number of simTracks that are "associated" to recoTracks
      int st(0);    	  //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) )
      unsigned sts(0);   //This counter counts the number of simTracks surviving the bunchcrossing cut
      unsigned asts(0);  //This counter counts the number of simTracks that are "associated" to recoTracks surviving the bunchcrossing cut
      for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){ //loop over TPs collection for tracking efficiency
	TrackingParticleRef tpr(TPCollectionHeff, i);
	TrackingParticle* tp=const_cast<TrackingParticle*>(tpr.get());  // why????
	TrackingParticle::Vector momentumTP;
	TrackingParticle::Point vertexTP;
	double dxySim(0);
	double dzSim(0);
	double dR=dR_tPCeff[i];

	//---------- THIS PART HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------
	//If the TrackingParticle is collison like, get the momentum and vertex at production state
	if(parametersDefiner=="LhcParametersDefinerForTP" || parametersDefiner=="hltLhcParametersDefinerForTP")
	  {
	    if(! tpSelector(*tp)) continue;
	    momentumTP = tp->momentum();
	    vertexTP = tp->vertex();
	    //Calcualte the impact parameters w.r.t. PCA
	    TrackingParticle::Vector momentum = parametersDefinerTP->momentum(event,setup,tpr);
	    TrackingParticle::Point vertex = parametersDefinerTP->vertex(event,setup,tpr);
	    dxySim = (-vertex.x()*sin(momentum.phi())+vertex.y()*cos(momentum.phi()));
	    dzSim = vertex.z() - (vertex.x()*momentum.x()+vertex.y()*momentum.y())/sqrt(momentum.perp2())
	      * momentum.z()/sqrt(momentum.perp2());
	  }
	//If the TrackingParticle is comics, get the momentum and vertex at PCA
	if(parametersDefiner=="CosmicParametersDefinerForTP")
	  {
	    if(! cosmictpSelector(tpr,&bs,event,setup)) continue;
	    momentumTP = parametersDefinerTP->momentum(event,setup,tpr);
	    vertexTP = parametersDefinerTP->vertex(event,setup,tpr);
	    dxySim = (-vertexTP.x()*sin(momentumTP.phi())+vertexTP.y()*cos(momentumTP.phi()));
	    dzSim = vertexTP.z() - (vertexTP.x()*momentumTP.x()+vertexTP.y()*momentumTP.y())/sqrt(momentumTP.perp2())
	      * momentumTP.z()/sqrt(momentumTP.perp2());
	  }
	//---------- THE PART ABOVE HAS TO BE CLEANED UP. THE PARAMETER DEFINER WAS NOT MEANT TO BE USED IN THIS WAY ----------

	st++;   //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) )

	// in the coming lines, histos are filled using as input
	// - momentumTP
	// - vertexTP
	// - dxySim
	// - dzSim

	histoProducerAlgo_->fill_generic_simTrack_histos(w,momentumTP,vertexTP, tp->eventId().bunchCrossing());


	// ##############################################
	// fill RecoAssociated SimTracks' histograms
	// ##############################################
	const reco::Track* matchedTrackPointer=0;
	if(simRecColl.find(tpr) != simRecColl.end()){
	  auto const & rt = simRecColl[tpr];
	  if (rt.size()!=0) {
	    ats++; //This counter counts the number of simTracks that have a recoTrack associated
	    // isRecoMatched = true; // UNUSED
	    matchedTrackPointer = rt.begin()->first.get();
	    edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st
					       << " with pt=" << sqrt(momentumTP.perp2())
					       << " associated with quality:" << rt.begin()->second <<"\n";
	  }
	}else{
	  edm::LogVerbatim("TrackValidator")
	    << "TrackingParticle #" << st
	    << " with pt,eta,phi: "
	    << sqrt(momentumTP.perp2()) << " , "
	    << momentumTP.eta() << " , "
	    << momentumTP.phi() << " , "
	    << " NOT associated to any reco::Track" << "\n";
	}




        int nSimHits = tp->numberOfTrackerHits();
	histoProducerAlgo_->fill_recoAssociated_simTrack_histos(w,*tp,momentumTP,vertexTP,dxySim,dzSim,nSimHits,matchedTrackPointer,puinfo.getPU_NumInteractions(), dR);
          sts++;
          if (matchedTrackPointer) asts++;




      } // End  for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){

      //if (st!=0) h_tracksSIM[w]->Fill(st);  // TO BE FIXED


      // ##############################################
      // fill recoTracks histograms (LOOP OVER TRACKS)
      // ##############################################
      edm::LogVerbatim("TrackValidator") << "\n# of reco::Tracks with "
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()
					 << ": " << trackCollection->size() << "\n";

      int sat(0); //This counter counts the number of recoTracks that are associated to SimTracks from Signal only
      int at(0); //This counter counts the number of recoTracks that are associated to SimTracks
      int rT(0); //This counter counts the number of recoTracks in general


      // dE/dx
      // at some point this could be generalized, with a vector of tags and a corresponding vector of Handles
      // I'm writing the interface such to take vectors of ValueMaps
      edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx1Handle;
      edm::Handle<edm::ValueMap<reco::DeDxData> > dEdx2Handle;
      std::vector<edm::ValueMap<reco::DeDxData> > v_dEdx;
      v_dEdx.clear();
      if (label[www].label()=="generalTracks") {
	try {
	  event.getByToken(m_dEdx1Tag, dEdx1Handle);
	  const edm::ValueMap<reco::DeDxData> dEdx1 = *dEdx1Handle.product();
	  event.getByToken(m_dEdx2Tag, dEdx2Handle);
	  const edm::ValueMap<reco::DeDxData> dEdx2 = *dEdx2Handle.product();
	  v_dEdx.push_back(dEdx1);
	  v_dEdx.push_back(dEdx2);
	} catch (cms::Exception e){
	  LogTrace("TrackValidator") << "exception found: " << e.what() << "\n";
	}
      }
      //end dE/dx


      //calculate dR for tracks
      float dR_trk[trackCollection->size()];
      int i=0;
      float etaL[trackCollectionForDrCalculation->size()];
      float phiL[trackCollectionForDrCalculation->size()];
      for (auto const & track2 : *trackCollectionForDrCalculation) {
         auto  && p = track2.momentum();
         etaL[i] = etaFromXYZ(p.x(),p.y(),p.z());
         phiL[i] = atan2f(p.y(),p.x());
         ++i;
      }
      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){
	auto const &  track = (*trackCollection)[i];
	auto dR = std::numeric_limits<float>::max();
        auto  && p = track.momentum();
        float eta = etaFromXYZ(p.x(),p.y(),p.z());
        float phi = atan2f(p.y(),p.x());
	for(View<Track>::size_type j=0; j<trackCollectionForDrCalculation->size(); ++j){
	  auto dR_tmp = reco::deltaR2(eta, phi, etaL[j], phiL[j]);
	  if ( (dR_tmp<dR) & (dR_tmp>std::numeric_limits<float>::min())) dR=dR_tmp;
	}
	dR_trk[i] = std::sqrt(dR);
      }

      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){

	RefToBase<Track> track(trackCollection, i);
	rT++;
 
        std::remove_reference<decltype(recSimColl[track])>::type dummyTP;
        
	bool isSigSimMatched(false);
	bool isSimMatched(false);
        bool isChargeMatched(true);
        int numAssocRecoTracks = 0;
	int nSimHits = 0;
	double sharedFraction = 0.;
	auto const & tp = (recSimColl.find(track) != recSimColl.end()) ? recSimColl[track] : dummyTP;
	
	if (!tp.empty()) {
	    nSimHits = tp[0].first->numberOfTrackerHits();
            sharedFraction = tp[0].second;
	    isSimMatched = true;
            if (tp[0].first->charge() != track->charge()) isChargeMatched = false;
            if(simRecColl.find(tp[0].first) != simRecColl.end()) numAssocRecoTracks = simRecColl[tp[0].first].size();
	    at++;
	    for (unsigned int tp_ite=0;tp_ite<tp.size();++tp_ite){
              TrackingParticle trackpart = *(tp[tp_ite].first);
	      if ((trackpart.eventId().event() == 0) && (trackpart.eventId().bunchCrossing() == 0)){
	      	isSigSimMatched = true;
		sat++;
		break;
	      }
            }
	    edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
					       << " associated with quality:" << tp.begin()->second <<"\n";
	} else {
	  edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
					     << " NOT associated to any TrackingParticle" << "\n";
	}

	double dR=dR_trk[i];
	histoProducerAlgo_->fill_generic_recoTrack_histos(w,*track,bs.position(),isSimMatched,isSigSimMatched, isChargeMatched, numAssocRecoTracks, puinfo.getPU_NumInteractions(), nSimHits, sharedFraction,dR);

	// dE/dx
	//	reco::TrackRef track2  = reco::TrackRef( trackCollection, i );
	if (v_dEdx.size() > 0) histoProducerAlgo_->fill_dedx_recoTrack_histos(w,track, v_dEdx);
	//if (v_dEdx.size() > 0) histoProducerAlgo_->fill_dedx_recoTrack_histos(track2, v_dEdx);


	//Fill other histos
 	//try{ //Is this really necessary ????

	if (tp.size()==0) continue;

	histoProducerAlgo_->fill_simAssociated_recoTrack_histos(w,*track);

	TrackingParticleRef tpr = tp.begin()->first;

	/* TO BE FIXED LATER
	if (associators[ww]=="trackAssociatorByChi2"){
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
	TrackingParticle::Vector momentumTP = parametersDefinerTP->momentum(event,setup,tpr);
	TrackingParticle::Point vertexTP = parametersDefinerTP->vertex(event,setup,tpr);
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
					 << "Total Associated (simToReco): " << ats << "\n"
					 << "Total Reconstructed: " << rT << "\n"
					 << "Total Associated (recoToSim): " << at << "\n"
					 << "Total Fakes: " << rT-at << "\n";

      w++;
    } // End of  for (unsigned int www=0;www<label.size();www++){
  } //END of for (unsigned int ww=0;ww<associators.size();ww++){

}

void MultiTrackValidator::endRun(Run const&, EventSetup const&) {
  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      if(!skipHistoFit && runStandalone)	histoProducerAlgo_->finalHistoFits(w);
      if (runStandalone) histoProducerAlgo_->fillProfileHistosFromVectors(w);
      w++;
    }
  }
  //if ( out.size() != 0 && dbe_ ) dbe_->save(out);
}



