#include "Validation/RecoTrack/interface/TrackerSeedValidator.h"
#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleNumberOfLayers.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimTracker/TrackAssociation/plugins/ParametersDefinerForTPESProducer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <TF1.h>

using namespace std;
using namespace edm;

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;

TrackerSeedValidator::TrackerSeedValidator(const edm::ParameterSet& pset):
  MultiTrackValidatorBase(pset, consumesCollector(),true),
  histoProducerAlgo_(std::make_unique<MTVHistoProducerAlgoForTracker>(pset.getParameter<ParameterSet>("histoProducerAlgoBlock"), consumesCollector())) {
  dirName_ = pset.getParameter<std::string>("dirName");

  tpSelector = TrackingParticleSelector(pset.getParameter<double>("ptMinTP"),
					pset.getParameter<double>("minRapidityTP"),
					pset.getParameter<double>("maxRapidityTP"),
					pset.getParameter<double>("tipTP"),
					pset.getParameter<double>("lipTP"),
					pset.getParameter<int>("minHitTP"),
					pset.getParameter<bool>("signalOnlyTP"),
					pset.getParameter<bool>("intimeOnlyTP"),
					pset.getParameter<bool>("chargedOnlyTP"),
					pset.getParameter<bool>("stableOnlyTP"),
					pset.getParameter<std::vector<int> >("pdgIdTP"));

  builderName = pset.getParameter<std::string>("TTRHBuilder");

  for (auto const& associator: associators) {
    associatorTokens.push_back(consumes<reco::TrackToTrackingParticleAssociator>(associator));
  }
}

TrackerSeedValidator::~TrackerSeedValidator() {}

void TrackerSeedValidator::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const& setup) {
  {
    ibook.cd();
    ibook.setCurrentFolder(dirName_ + "simulation");

    //Booking histograms concerning with simulated tracks
    histoProducerAlgo_->bookSimHistos(ibook);
  }

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
      //      if (dirName.find("Seeds")<dirName.length()){
      //    dirName.replace(dirName.find("Seeds"),6,"");
      //      }
      string assoc= associators[ww].label();;
      if (assoc.find("Track")<assoc.length()){
    assoc.replace(assoc.find("Track"),5,"");
      }
      dirName+=assoc;
      std::replace(dirName.begin(), dirName.end(), ':', '_');

      ibook.setCurrentFolder(dirName.c_str());

      //Booking histograms concerning with reconstructed tracks
      histoProducerAlgo_->bookSimTrackHistos(ibook);
      histoProducerAlgo_->bookRecoHistos(ibook);
    }//end loop www
  }// end loop ww
}


void TrackerSeedValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){

  edm::LogInfo("TrackValidator") << "\n====================================================" << "\n"
				 << "Analyzing new event" << "\n"
				 << "====================================================\n" << "\n";

  edm::ESHandle<ParametersDefinerForTP> parametersDefinerTP;
  setup.get<TrackAssociatorRecord>().get(parametersDefiner,parametersDefinerTP);

  edm::ESHandle<TrackerTopology> httopo;
  setup.get<TrackerTopologyRcd>().get(httopo);
  const TrackerTopology& ttopo = *httopo;

  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
  setup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);

  edm::ESHandle<MagneticField> theMF;
  setup.get<IdealMagneticFieldRecord>().get(theMF);

  edm::Handle<TrackingParticleCollection>  TPCollectionHeff ;
  event.getByToken(label_tp_effic,TPCollectionHeff);
  const TrackingParticleCollection tPCeff = *(TPCollectionHeff.product());

  edm::Handle<TrackingParticleCollection>  TPCollectionHfake ;
  event.getByToken(label_tp_fake,TPCollectionHfake);
  const TrackingParticleCollection tPCfake = *(TPCollectionHfake.product());

  if (tPCeff.size()==0) {edm::LogInfo("TrackValidator") << "TP Collection for efficiency studies has size = 0! Skipping Event." ; return;}
  if (tPCfake.size()==0) {edm::LogInfo("TrackValidator") << "TP Collection for fake rate studies has size = 0! Skipping Event." ; return;}

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  event.getByToken(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;

  edm::Handle< vector<PileupSummaryInfo> > puinfoH;
  event.getByToken(label_pileupinfo,puinfoH);
  PileupSummaryInfo puinfo;

  for (unsigned int puinfo_ite=0;puinfo_ite<(*puinfoH).size();++puinfo_ite){
    if ((*puinfoH)[puinfo_ite].getBunchCrossing()==0){
      puinfo=(*puinfoH)[puinfo_ite];
      break;
    }
  }

  // Calculate the number of 3D layers for TPs
  //
  // I would have preferred to produce the ValueMap to Event and read
  // it from there, but there would have been quite some number of
  // knock-on effects, and again the fact that we take two TP
  // collections do not support Ref<TP>'s would have complicated that.
  //
  // In principle we could use the SimHitTPAssociationList read above
  // for parametersDefinerIsCosmic_, but since we don't currently
  // support Ref<TP>s, we can't in general use it since eff/fake TP
  // collections can, in general, be different.
  TrackingParticleNumberOfLayers tpNumberOfLayersAlgo(event, simHitTokens_);
  auto nlayers_tPCeff_ptrs = tpNumberOfLayersAlgo.calculate(TPCollectionHeff, setup);
  const auto& nLayers_tPCeff = *(std::get<TrackingParticleNumberOfLayers::nTrackerLayers>(nlayers_tPCeff_ptrs));
  const auto& nPixelLayers_tPCeff = *(std::get<TrackingParticleNumberOfLayers::nPixelLayers>(nlayers_tPCeff_ptrs));
  const auto& nStripMonoAndStereoLayers_tPCeff = *(std::get<TrackingParticleNumberOfLayers::nStripMonoAndStereoLayers>(nlayers_tPCeff_ptrs));

  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
    event.getByToken(associatorTokens[ww], theAssociator);
    const reco::TrackToTrackingParticleAssociator *associator = theAssociator.product();

    for (unsigned int www=0;www<label.size();www++){
      edm::LogVerbatim("TrackValidator") << "Analyzing "
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()<<" with "
					 << associators[ww] <<"\n";
      //
      //get collections from the event
      //
      edm::Handle<edm::View<TrajectorySeed> > seedCollection;
      event.getByToken(labelTokenSeed[www], seedCollection);
      if (seedCollection->size()==0) {
	edm::LogInfo("TrackValidator") << "SeedCollection size = 0!" ;
	continue;
      }

      //associate seeds
      LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
      reco::RecoToSimCollectionSeed recSimColl=associator->associateRecoToSim(seedCollection,
                                                                              TPCollectionHfake);
      LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
      reco::SimToRecoCollectionSeed simRecColl=associator->associateSimToReco(seedCollection,
                                                                              TPCollectionHeff);

      //
      //fill simulation histograms
      //compute number of seeds per eta interval
      //
      edm::LogVerbatim("TrackValidator") << "\n# of TrackingParticles: " << tPCeff.size() << "\n";
      int ats(0);  	  //This counter counts the number of simTracks that are "associated" to recoTracks
      int st(0);    	  //This counter counts the number of simulated tracks passing the MTV selection (i.e. tpSelector(tp) )
      unsigned sts(0);   //This counter counts the number of simTracks surviving the bunchcrossing cut
      unsigned asts(0);  //This counter counts the number of simTracks that are "associated" to recoTracks surviving the bunchcrossing cut
      for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
	TrackingParticleRef tp(TPCollectionHeff, i);

	if (tp->charge()==0) continue;

	if(! tpSelector(*tp)) continue;

	TrackingParticle::Vector momentumTP = tp->momentum();
	TrackingParticle::Point vertexTP = tp->vertex();
	//Calcualte the impact parameters w.r.t. PCA
	TrackingParticle::Vector momentum = parametersDefinerTP->momentum(event,setup,tp);
	TrackingParticle::Point vertex = parametersDefinerTP->vertex(event,setup,tp);
	double dxySim = (-vertex.x()*sin(momentum.phi())+vertex.y()*cos(momentum.phi()));
	double dzSim = vertex.z() - (vertex.x()*momentum.x()+vertex.y()*momentum.y())/sqrt(momentum.perp2())
	  * momentum.z()/sqrt(momentum.perp2());

        if(tp->eventId().bunchCrossing() == 0) {
          st++;
        }

        if(w == 0)
          histoProducerAlgo_->fill_generic_simTrack_histos(momentumTP,vertexTP, tp->eventId().bunchCrossing());

	const TrajectorySeed* matchedSeedPointer=0;
	std::vector<std::pair<edm::RefToBase<TrajectorySeed>, double> > rt;
	if(simRecColl.find(tp) != simRecColl.end()){
	  rt = simRecColl[tp];
	  if (rt.size()!=0) {
	    ats++;
	    matchedSeedPointer = rt.begin()->first.get();
	    edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st
					       << " with pt=" << sqrt(tp->momentum().perp2())
					       << " associated with quality:" << rt.begin()->second <<"\n";
	  }
	}else{
	  edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st
					     << " with pt=" << sqrt(tp->momentum().perp2())
					     << " NOT associated to any TrajectorySeed" << "\n";
	}

        int nSimHits = tp->numberOfTrackerHits();
        int nSimLayers = nLayers_tPCeff[tp];
        int nSimPixelLayers = nPixelLayers_tPCeff[tp];
        int nSimStripMonoAndStereoLayers = nStripMonoAndStereoLayers_tPCeff[tp];

	//fixme convert seed into track
	reco::Track* matchedTrackPointer = 0;
	if (matchedSeedPointer) {
	  TSCBLBuilderNoMaterial tscblBuilder;
	  TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(matchedSeedPointer->recHits().second-1));
	  TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( matchedSeedPointer->startingState(), recHit->surface(), theMF.product());
	  TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithm
	  if(!(tsAtClosestApproachSeed.isValid())){
	    edm::LogVerbatim("SeedValidator")<<"TrajectoryStateClosestToBeamLine not valid";
	    continue;
	  }
	  const reco::TrackBase::Point vSeed1(tsAtClosestApproachSeed.trackStateAtPCA().position().x(),
					      tsAtClosestApproachSeed.trackStateAtPCA().position().y(),
					      tsAtClosestApproachSeed.trackStateAtPCA().position().z());
	  const reco::TrackBase::Vector pSeed(tsAtClosestApproachSeed.trackStateAtPCA().momentum().x(),
					      tsAtClosestApproachSeed.trackStateAtPCA().momentum().y(),
					      tsAtClosestApproachSeed.trackStateAtPCA().momentum().z());
	  //GlobalPoint vSeed(vSeed1.x()-bs.x0(),vSeed1.y()-bs.y0(),vSeed1.z()-bs.z0());
	  PerigeeTrajectoryError seedPerigeeErrors = PerigeeConversions::ftsToPerigeeError(tsAtClosestApproachSeed.trackStateAtPCA());
	  matchedTrackPointer = new reco::Track(0.,0., vSeed1, pSeed, 1, seedPerigeeErrors.covarianceMatrix());
	  matchedTrackPointer->appendHits(matchedSeedPointer->recHits().first,matchedSeedPointer->recHits().second, ttopo);
	}

	double dR=0;//fixme: plots vs dR not implemented for now
	histoProducerAlgo_->fill_recoAssociated_simTrack_histos(w,*tp,tp->momentum(),tp->vertex(),dxySim,dzSim,0,0,nSimHits,nSimLayers,nSimPixelLayers,nSimStripMonoAndStereoLayers,
								matchedTrackPointer,puinfo.getPU_NumInteractions(),dR, nullptr);

	sts++;
	if (matchedTrackPointer) asts++;

      } // End  for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){

      if(w == 0)
        histoProducerAlgo_->fill_simTrackBased_histos(st);

      //
      //fill reconstructed seed histograms
      //
      edm::LogVerbatim("TrackValidator") << "\n# of TrajectorySeeds with "
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()
					 << ": " << seedCollection->size() << "\n";
      int sat(0); //This counter counts the number of recoTracks that are associated to SimTracks from Signal only
      int at(0); //This counter counts the number of recoTracks that are associated to SimTracks
      int rT(0); //This counter counts the number of recoTracks in general

      TSCBLBuilderNoMaterial tscblBuilder;
      for(TrajectorySeedCollection::size_type i=0; i<seedCollection->size(); ++i){
	edm::RefToBase<TrajectorySeed> seed(seedCollection, i);
	rT++;

	//get parameters and errors from the seed state
	TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(seed->recHits().second-1));
	TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( seed->startingState(), recHit->surface(), theMF.product());
	TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithm
	if(!(tsAtClosestApproachSeed.isValid())){
	  edm::LogVerbatim("SeedValidator")<<"TrajectoryStateClosestToBeamLine not valid";
	  continue;
	    }
	const reco::TrackBase::Point vSeed1(tsAtClosestApproachSeed.trackStateAtPCA().position().x(),
					    tsAtClosestApproachSeed.trackStateAtPCA().position().y(),
					    tsAtClosestApproachSeed.trackStateAtPCA().position().z());
	const reco::TrackBase::Vector pSeed(tsAtClosestApproachSeed.trackStateAtPCA().momentum().x(),
					    tsAtClosestApproachSeed.trackStateAtPCA().momentum().y(),
					    tsAtClosestApproachSeed.trackStateAtPCA().momentum().z());
	//GlobalPoint vSeed(vSeed1.x()-bs.x0(),vSeed1.y()-bs.y0(),vSeed1.z()-bs.z0());
	PerigeeTrajectoryError seedPerigeeErrors = PerigeeConversions::ftsToPerigeeError(tsAtClosestApproachSeed.trackStateAtPCA());

	//fixme
	reco::Track* trackFromSeed = new reco::Track(0.,0., vSeed1, pSeed, 1, seedPerigeeErrors.covarianceMatrix());
	trackFromSeed->appendHits(seed->recHits().first,seed->recHits().second, ttopo);

	bool isSigSimMatched(false);
	bool isSimMatched(false);
	bool isChargeMatched(true);
	int numAssocSeeds = 0;
	int nSimHits = 0;
	double sharedFraction = 0.;
	std::vector<std::pair<TrackingParticleRef, double> > tp;
	if(recSimColl.find(seed) != recSimColl.end()) {
	  tp = recSimColl[seed];
	  if (tp.size()!=0) {

            nSimHits = tp[0].first->numberOfTrackerHits();
            sharedFraction = tp[0].second;
	    isSimMatched = true;
	    if (tp[0].first->charge() != seed->startingState().parameters().charge()) isChargeMatched = false;
	    if(simRecColl.find(tp[0].first) != simRecColl.end()) numAssocSeeds = simRecColl[tp[0].first].size();
	    //std::cout << numAssocRecoTracks << std::endl;

	    at++;

	    for (unsigned int tp_ite=0;tp_ite<tp.size();++tp_ite){
              TrackingParticle trackpart = *(tp[tp_ite].first);
	      if ((trackpart.eventId().event() == 0) && (trackpart.eventId().bunchCrossing() == 0)){
	      	isSigSimMatched = true;
		sat++;
		break;
	      }
            }


	    edm::LogVerbatim("SeedValidator") << "TrajectorySeed #" << rT << " associated with quality:" << tp.begin()->second <<"\n";
	  }
	} else {
	  edm::LogVerbatim("SeedValidator") << "TrajectorySeed #" << rT << " NOT associated to any TrackingParticle" << "\n";
	}

	double dR = 0.;//fixme: plots vs dR not implemented for now
	histoProducerAlgo_->fill_generic_recoTrack_histos(w,*trackFromSeed,bs.position(), nullptr, isSimMatched,isSigSimMatched,
							  isChargeMatched, numAssocSeeds, 
							  puinfo.getPU_NumInteractions(),
							  nSimHits, sharedFraction, dR);

	//Fill other histos
 	try{
	  if (tp.size()==0) continue;

	  histoProducerAlgo_->fill_simAssociated_recoTrack_histos(w,*trackFromSeed);

	  TrackingParticleRef tpr = tp.begin()->first;

	  //compute tracking particle parameters at point of closest approach to the beamline
	  TrackingParticle::Vector momentumTP = parametersDefinerTP->momentum(event,setup,tpr);
	  TrackingParticle::Point vertexTP = parametersDefinerTP->vertex(event,setup,tpr);

	  // 	  LogTrace("SeedValidatorTEST") << "assocChi2=" << tp.begin()->second << "\n"
	  // 					 << "" <<  "\n"
	  // 					 << "ptREC=" << ptSeed << "\n"
	  // 					 << "etaREC=" << etaSeed << "\n"
	  // 					 << "qoverpREC=" << qoverpSeed << "\n"
	  // 					 << "dxyREC=" << dxySeed << "\n"
	  // 					 << "dzREC=" << dzSeed << "\n"
	  // 					 << "thetaREC=" << thetaSeed << "\n"
	  // 					 << "phiREC=" << phiSeed << "\n"
	  // 					 << "" <<  "\n"
	  // 					 << "qoverpError()=" << qoverpErrorSeed << "\n"
	  // 					 << "dxyError()=" << dxyErrorSeed << "\n"
	  // 					 << "dzError()=" << dzErrorSeed << "\n"
	  // 					 << "thetaError()=" << lambdaErrorSeed << "\n"
	  // 					 << "phiError()=" << phiErrorSeed << "\n"
	  // 					 << "" <<  "\n"
	  // 					 << "ptSIM=" << sqrt(assocTrack->momentum().perp2()) << "\n"
	  // 					 << "etaSIM=" << assocTrack->momentum().Eta() << "\n"
	  // 					 << "qoverpSIM=" << qoverpSim << "\n"
	  // 					 << "dxySIM=" << dxySim << "\n"
	  // 					 << "dzSIM=" << dzSim << "\n"
	  // 					 << "thetaSIM=" << M_PI/2-lambdaSim << "\n"
	  // 					 << "phiSIM=" << phiSim << "\n"
	  // 					 << "" << "\n"
	  // 					 << "contrib_Qoverp=" << contrib_Qoverp << "\n"
	  // 					 << "contrib_dxy=" << contrib_dxy << "\n"
	  // 					 << "contrib_dz=" << contrib_dz << "\n"
	  // 					 << "contrib_theta=" << contrib_theta << "\n"
	  // 					 << "contrib_phi=" << contrib_phi << "\n"
	  // 					 << "" << "\n"
	  // 					 <<"chi2PULL="<<contrib_Qoverp+contrib_dxy+contrib_dz+contrib_theta+contrib_phi<<"\n";

	  histoProducerAlgo_->fill_ResoAndPull_recoTrack_histos(w,momentumTP,vertexTP,tpr->charge(),
								*trackFromSeed,bs.position());


	} catch (cms::Exception e){
	  LogTrace("SeedValidator") << "exception found: " << e.what() << "\n";
	}
      }// End of for(TrajectorySeedCollection::size_type i=0; i<seedCollection->size(); ++i)

      histoProducerAlgo_->fill_trackBased_histos(w,at,rT,st);

      edm::LogVerbatim("SeedValidator") << "Total Simulated: " << st << "\n"
					 << "Total Associated (simToReco): " << ats << "\n"
					 << "Total Reconstructed: " << rT << "\n"
					 << "Total Associated (recoToSim): " << at << "\n"
					 << "Total Fakes: " << rT-at << "\n";
      w++;
    }
  }
}
