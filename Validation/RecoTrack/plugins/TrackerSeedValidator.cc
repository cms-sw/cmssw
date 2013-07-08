/*
#include "Validation/RecoTrack/interface/TrackerSeedValidator.h"
#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/QuickTrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include <TF1.h>
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

using namespace std;
using namespace edm;

void TrackerSeedValidator::beginRun(edm::Run const&, edm::EventSetup const& setup) {
  setup.get<IdealMagneticFieldRecord>().get(theMF);  
  setup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);

  int j=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){

      dbe_->cd();
      InputTag algo = label[www];
      string dirName="Tracking/Seed/";
      if (algo.process()!="")
	dirName+=algo.process()+"_";
      if(algo.label()!="")
	dirName+=algo.label()+"_";
      if(algo.instance()!="")
	dirName+=algo.instance()+"_";      
      //      if (dirName.find("Seeds")<dirName.length()){
      //	dirName.replace(dirName.find("Seeds"),6,"");
      //      }
      string assoc= associators[ww];
      if (assoc.find("Track")<assoc.length()){
	assoc.replace(assoc.find("Track"),5,"");
      }
      dirName+=assoc;
      dbe_->setCurrentFolder(dirName.c_str());

      setUpVectors();

      dbe_->goUp();
      string subDirName = dirName + "/simulation";
      dbe_->setCurrentFolder(subDirName.c_str());
      h_ptSIM.push_back( dbe_->book1D("ptSIM", "generated p_{t}", 5500, 0, 110 ) );
      h_etaSIM.push_back( dbe_->book1D("etaSIM", "generated pseudorapidity", 500, -2.5, 2.5 ) );
      h_tracksSIM.push_back( dbe_->book1D("tracksSIM","number of simluated tracks",100,-0.5,99.5) );
      h_vertposSIM.push_back( dbe_->book1D("vertposSIM","Transverse position of sim vertices",1000,-0.5,10000.5) );
      
      dbe_->cd();
      dbe_->setCurrentFolder(dirName.c_str());
      h_tracks.push_back( dbe_->book1D("seeds","number of reconstructed seeds",20,-0.5,19.5) );
      h_fakes.push_back( dbe_->book1D("fakes","number of fake reco seeds",20,-0.5,19.5) );
      h_charge.push_back( dbe_->book1D("charge","charge",3,-1.5,1.5) );
      h_hits.push_back( dbe_->book1D("hits", "number of hits per seed", 30, -0.5, 29.5 ) );

      h_effic.push_back( dbe_->book1D("effic","efficiency vs #eta",nint,min,max) );
      h_efficPt.push_back( dbe_->book1D("efficPt","efficiency vs pT",nintpT,minpT,maxpT) );
      h_fakerate.push_back( dbe_->book1D("fakerate","fake rate vs #eta",nint,min,max) );
      h_fakeratePt.push_back( dbe_->book1D("fakeratePt","fake rate vs pT",nintpT,minpT,maxpT) );
      h_effic_vs_hit.push_back( dbe_->book1D("effic_vs_hit","effic vs hit",nintHit,minHit,maxHit) );
      h_fake_vs_hit.push_back( dbe_->book1D("fakerate_vs_hit","fake rate vs hit",nintHit,minHit,maxHit) );

      h_recoeta.push_back( dbe_->book1D("num_reco_eta","N of reco seed vs eta",nint,min,max) );
      h_assoceta.push_back( dbe_->book1D("num_assoc(simToReco)_eta","N of associated seeds (simToReco) vs eta",nint,min,max) );
      h_assoc2eta.push_back( dbe_->book1D("num_assoc(recoToSim)_eta","N of associated (recoToSim) seeds vs eta",nint,min,max) );
      h_simuleta.push_back( dbe_->book1D("num_simul_eta","N of simulated tracks vs eta",nint,min,max) );
      h_recopT.push_back( dbe_->book1D("num_reco_pT","N of reco seed vs pT",nintpT,minpT,maxpT) );
      h_assocpT.push_back( dbe_->book1D("num_assoc(simToReco)_pT","N of associated seeds (simToReco) vs pT",nintpT,minpT,maxpT) );
      h_assoc2pT.push_back( dbe_->book1D("num_assoc(recoToSim)_pT","N of associated (recoToSim) seeds vs pT",nintpT,minpT,maxpT) );
      h_simulpT.push_back( dbe_->book1D("num_simul_pT","N of simulated tracks vs pT",nintpT,minpT,maxpT) );
      
      h_eta.push_back( dbe_->book1D("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
      h_pt.push_back( dbe_->book1D("pullPt", "pull of p_{t}", 100, -10, 10 ) );
      h_pullTheta.push_back( dbe_->book1D("pullTheta","pull of #theta parameter",250,-25,25) );
      h_pullPhi.push_back( dbe_->book1D("pullPhi","pull of #phi parameter",250,-25,25) );
      h_pullDxy.push_back( dbe_->book1D("pullDxy","pull of dxy parameter",250,-25,25) );
      h_pullDz.push_back( dbe_->book1D("pullDz","pull of dz parameter",250,-25,25) );
      h_pullQoverp.push_back( dbe_->book1D("pullQoverp","pull of qoverp parameter",250,-25,25) );
      
      if (associators[ww]=="quickTrackAssociatorByHits"){
	h_assocFraction.push_back( dbe_->book1D("assocFraction","fraction of shared hits",200,0,2) );
	h_assocSharedHit.push_back(dbe_->book1D("assocSharedHit","number of shared hits",20,0,20));
      }

      //chi2_vs_nhits.push_back( dbe_->book2D("chi2_vs_nhits","#chi^{2} vs nhits",25,0,25,100,0,10) );
      nrec_vs_nsim.push_back( dbe_->book2D("nrec_vs_nsim","nrec vs nsim",20,-0.5,19.5,20,-0.5,19.5) );

      nhits_vs_eta.push_back( dbe_->book2D("nhits_vs_eta","nhits vs eta",nint,min,max,nintHit,minHit,maxHit) );
      h_hits_eta.push_back( dbe_->bookProfile("hits_eta","mean #hits vs eta",nint,min,max,nintHit,minHit,maxHit) );

      if(useLogPt){
	BinLogX(h_efficPt[j]->getTH1F());
	BinLogX(h_fakeratePt[j]->getTH1F());
	BinLogX(h_recopT[j]->getTH1F());
	BinLogX(h_assocpT[j]->getTH1F());
	BinLogX(h_assoc2pT[j]->getTH1F());
	BinLogX(h_simulpT[j]->getTH1F());
      }      
      j++;
    }
  }
  edm::ESHandle<TrackAssociatorBase> theAssociator;
  for (unsigned int w=0;w<associators.size();w++) {
    setup.get<TrackAssociatorRecord>().get(associators[w],theAssociator);
    associator.push_back( theAssociator.product() );
  }
}

void TrackerSeedValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){

  edm::LogInfo("TrackValidator") << "\n====================================================" << "\n"
				 << "Analyzing new event" << "\n"
				 << "====================================================\n" << "\n";
  
  edm::Handle<TrackingParticleCollection>  TPCollectionHeff ;
  event.getByLabel(label_tp_effic,TPCollectionHeff);
  const TrackingParticleCollection tPCeff = *(TPCollectionHeff.product());
  
  edm::Handle<TrackingParticleCollection>  TPCollectionHfake ;
  event.getByLabel(label_tp_fake,TPCollectionHfake);
  const TrackingParticleCollection tPCfake = *(TPCollectionHfake.product());

  if (tPCeff.size()==0) {edm::LogInfo("TrackValidator") << "TP Collection for efficiency studies has size = 0! Skipping Event." ; return;}
  if (tPCfake.size()==0) {edm::LogInfo("TrackValidator") << "TP Collection for fake rate studies has size = 0! Skipping Event." ; return;}

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  event.getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      
  
  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      edm::LogVerbatim("TrackValidator") << "Analyzing " 
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()<<" with "
					 << associators[ww].c_str() <<"\n";
      //
      //get collections from the event
      //
      edm::Handle<edm::View<TrajectorySeed> > seedCollection;
      event.getByLabel(label[www], seedCollection);
      if (seedCollection->size()==0) {
	edm::LogInfo("TrackValidator") << "SeedCollection size = 0!" ; 
	continue;
      }
 
      //associate seeds
      LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
      reco::RecoToSimCollectionSeed recSimColl=associator[ww]->associateRecoToSim(seedCollection,
										  TPCollectionHfake,
										  &event);
      LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
      reco::SimToRecoCollectionSeed simRecColl=associator[ww]->associateSimToReco(seedCollection,
										  TPCollectionHeff, 
										  &event);
      
      //
      //fill simulation histograms
      //compute number of seeds per eta interval
      //
      edm::LogVerbatim("TrackValidator") << "\n# of TrackingParticles: " << tPCeff.size() << "\n";
      int ats = 0;
      int st=0;
      for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
	TrackingParticleRef tp(TPCollectionHeff, i);
	if (tp->charge()==0) continue;
	st++;
	h_ptSIM[w]->Fill(sqrt(tp->momentum().perp2()));
	h_etaSIM[w]->Fill(tp->momentum().eta());
	h_vertposSIM[w]->Fill(sqrt(tp->vertex().perp2()));

	std::vector<std::pair<edm::RefToBase<TrajectorySeed>, double> > rt;
	if(simRecColl.find(tp) != simRecColl.end()){
	  rt = simRecColl[tp];
	  if (rt.size()!=0) {
	    ats++;
	    edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st 
					       << " with pt=" << sqrt(tp->momentum().perp2()) 
					       << " associated with quality:" << rt.begin()->second <<"\n";
	  }
	}else{
	  edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st
					     << " with pt=" << sqrt(tp->momentum().perp2())
					     << " NOT associated to any TrajectorySeed" << "\n";
	}
	double tpeta=getEta(tp->momentum().eta());
	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (tpeta>etaintervals[w][f]&&
	      tpeta<etaintervals[w][f+1]) {
	    totSIMeta[w][f]++;
	    if (rt.size()!=0) {
	      totASSeta[w][f]++;
	    }
	  }
	} // END for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	double tppt=getPt(sqrt(tp->momentum().perp2()));
	for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
          if (tppt>pTintervals[w][f]&&
              tppt<pTintervals[w][f+1]) {
            totSIMpT[w][f]++;
	    if (rt.size()!=0) {
	      totASSpT[w][f]++;
	    }
	  }
	} // END for (unsigned int f=0; f<pTintervals[w].size()-1; f++){

	totSIM_hit[w][tp->matchedHit()]++;//FIXME
	if (rt.size()!=0) totASS_hit[w][tp->matchedHit()]++;
      }
      if (st!=0) h_tracksSIM[w]->Fill(st);
      

      //
      //fill reconstructed seed histograms
      // 
      edm::LogVerbatim("TrackValidator") << "\n# of TrajectorySeeds with "
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()
					 << ": " << seedCollection->size() << "\n";
      int at=0;
      int rT=0;
      
      TSCBLBuilderNoMaterial tscblBuilder;
      PerigeeConversions tspConverter;
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
	GlobalPoint vSeed1 = tsAtClosestApproachSeed.trackStateAtPCA().position();
	GlobalVector pSeed = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
	GlobalPoint vSeed(vSeed1.x()-bs.x0(),vSeed1.y()-bs.y0(),vSeed1.z()-bs.z0());

	double etaSeed = state.globalMomentum().eta();
	double ptSeed = sqrt(state.globalMomentum().perp2());
	double pmSeed = sqrt(state.globalMomentum().mag2());
	double pzSeed = state.globalMomentum().z();
	double numberOfHitsSeed = seed->recHits().second-seed->recHits().first;
	double qoverpSeed = tsAtClosestApproachSeed.trackStateAtPCA().charge()/pSeed.mag();
	double thetaSeed  = pSeed.theta();
	double lambdaSeed = M_PI/2-thetaSeed;
	double phiSeed    = pSeed.phi();
	double dxySeed    = (-vSeed.x()*sin(pSeed.phi())+vSeed.y()*cos(pSeed.phi()));
	double dzSeed     = vSeed.z() - (vSeed.x()*pSeed.x()+vSeed.y()*pSeed.y())/pSeed.perp() * pSeed.z()/pSeed.perp();

	PerigeeTrajectoryError seedPerigeeErrors = tspConverter.ftsToPerigeeError(tsAtClosestApproachSeed.trackStateAtPCA());
	double qoverpErrorSeed = tsAtClosestApproachSeed.trackStateAtPCA().curvilinearError().matrix().At(0,0);
	double lambdaErrorSeed = seedPerigeeErrors.thetaError();//=theta error
	double phiErrorSeed = seedPerigeeErrors.phiError();
	double dxyErrorSeed = seedPerigeeErrors.transverseImpactParameterError();
	double dzErrorSeed = seedPerigeeErrors.longitudinalImpactParameterError();
	double ptErrorSeed = (state.charge()!=0) ?  sqrt(
	       ptSeed*ptSeed*pmSeed*pmSeed/state.charge()/state.charge() * tsAtClosestApproachSeed.trackStateAtPCA().curvilinearError().matrix().At(0,0)
	       + 2*ptSeed*pmSeed/state.charge()*pzSeed * tsAtClosestApproachSeed.trackStateAtPCA().curvilinearError().matrix().At(0,1)
	       + pzSeed*pzSeed * tsAtClosestApproachSeed.trackStateAtPCA().curvilinearError().matrix().At(1,1) ) : 1.e6;
	
	std::vector<std::pair<TrackingParticleRef, double> > tp;
	if(recSimColl.find(seed) != recSimColl.end()) {
	  tp = recSimColl[seed];
	  if (tp.size()!=0) {
	    at++;
	    edm::LogVerbatim("SeedValidator") << "TrajectorySeed #" << rT << " associated with quality:" << tp.begin()->second <<"\n";
	  }
	} else {
	  edm::LogVerbatim("SeedValidator") << "TrajectorySeed #" << rT << " NOT associated to any TrackingParticle" << "\n";		  
	}
	
	//Compute fake rate vs eta
	double seedeta=getEta(etaSeed);
	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (seedeta>etaintervals[w][f]&&
	      seedeta<etaintervals[w][f+1]) {
	    totRECeta[w][f]++; 
	    if (tp.size()!=0) {
	      totASS2eta[w][f]++;
	    }		
	  }
	}
	double seedpt=getPt(ptSeed);
	for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
	  if (seedpt>pTintervals[w][f]&&
	      seedpt<pTintervals[w][f+1]) {
	    totRECpT[w][f]++; 
	    if (tp.size()!=0) {
	      totASS2pT[w][f]++;
	    }	      
	  }
	}
 	totREC_hit[w][seed->nHits()]++;
	if (tp.size()!=0) totASS2_hit[w][seed->nHits()]++;
	
	//Fill other histos
 	try{
	  if (tp.size()==0) continue;
	
	  TrackingParticleRef tpr = tp.begin()->first;
	  const SimTrack * assocTrack = &(*tpr->g4Track_begin());
	
	  if (associators[ww]=="quickTrackAssociatorByHits"){
	    double fraction = tp.begin()->second;
	    h_assocFraction[www]->Fill(fraction);
	    h_assocSharedHit[www]->Fill(fraction*numberOfHitsSeed);
	  }
    
	  h_hits[w]->Fill(numberOfHitsSeed);
	  h_charge[w]->Fill( state.charge() );
	
	  //compute tracking particle parameters at point of closest approach to the beamline
	  edm::ESHandle<MagneticField> theMF;
	  setup.get<IdealMagneticFieldRecord>().get(theMF);
	  FreeTrajectoryState 
	    ftsAtProduction(GlobalPoint(tpr->vertex().x(),tpr->vertex().y(),tpr->vertex().z()),
			    GlobalVector(assocTrack->momentum().x(),assocTrack->momentum().y(),assocTrack->momentum().z()),
			    TrackCharge(tpr->charge()),
			    theMF.product());
	  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,bs);//as in TrackProducerAlgorithm
	  GlobalPoint v1 = tsAtClosestApproach.trackStateAtPCA().position();
	  GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
	  GlobalPoint v(v1.x()-bs.x0(),v1.y()-bs.y0(),v1.z()-bs.z0());

	  double qoverpSim = tsAtClosestApproach.trackStateAtPCA().charge()/p.mag();
	  double lambdaSim = M_PI/2-p.theta();
	  double phiSim    = p.phi();
	  double dxySim    = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
	  double dzSim     = v.z() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.perp();

	  // eta residue; pt, k, theta, phi, dxy, dz pulls
	  double qoverpPull=(qoverpSeed-qoverpSim)/qoverpErrorSeed;
	  double thetaPull=(lambdaSeed-lambdaSim)/lambdaErrorSeed;
	  double phiPull=(phiSeed-phiSim)/phiErrorSeed;
	  double dxyPull=(dxySeed-dxySim)/dxyErrorSeed;
	  double dzPull=(dzSeed-dzSim)/dzErrorSeed;

	  double contrib_Qoverp = ((qoverpSeed-qoverpSim)/qoverpErrorSeed)*
	    ((qoverpSeed-qoverpSim)/qoverpErrorSeed)/5;
	  double contrib_dxy = ((dxySeed-dxySim)/dxyErrorSeed)*((dxySeed-dxySim)/dxyErrorSeed)/5;
	  double contrib_dz = ((dzSeed-dzSim)/dzErrorSeed)*((dzSeed-dzSim)/dzErrorSeed)/5;
	  double contrib_theta = ((lambdaSeed-lambdaSim)/lambdaErrorSeed)*
	    ((lambdaSeed-lambdaSim)/lambdaErrorSeed)/5;
	  double contrib_phi = ((phiSeed-phiSim)/phiErrorSeed)*
	    ((phiSeed-phiSim)/phiErrorSeed)/5;
	  LogTrace("SeedValidatorTEST") << "assocChi2=" << tp.begin()->second << "\n"
					 << "" <<  "\n"
					 << "ptREC=" << ptSeed << "\n"
					 << "etaREC=" << etaSeed << "\n"
					 << "qoverpREC=" << qoverpSeed << "\n"
					 << "dxyREC=" << dxySeed << "\n"
					 << "dzREC=" << dzSeed << "\n"
					 << "thetaREC=" << thetaSeed << "\n"
					 << "phiREC=" << phiSeed << "\n"
					 << "" <<  "\n"
					 << "qoverpError()=" << qoverpErrorSeed << "\n"
					 << "dxyError()=" << dxyErrorSeed << "\n"
					 << "dzError()=" << dzErrorSeed << "\n"
					 << "thetaError()=" << lambdaErrorSeed << "\n"
					 << "phiError()=" << phiErrorSeed << "\n"
					 << "" <<  "\n"
					 << "ptSIM=" << sqrt(assocTrack->momentum().perp2()) << "\n"
					 << "etaSIM=" << assocTrack->momentum().Eta() << "\n"    
					 << "qoverpSIM=" << qoverpSim << "\n"
					 << "dxySIM=" << dxySim << "\n"
					 << "dzSIM=" << dzSim << "\n"
					 << "thetaSIM=" << M_PI/2-lambdaSim << "\n"
					 << "phiSIM=" << phiSim << "\n"
					 << "" << "\n"
					 << "contrib_Qoverp=" << contrib_Qoverp << "\n"
					 << "contrib_dxy=" << contrib_dxy << "\n"
					 << "contrib_dz=" << contrib_dz << "\n"
					 << "contrib_theta=" << contrib_theta << "\n"
					 << "contrib_phi=" << contrib_phi << "\n"
					 << "" << "\n"
					 <<"chi2PULL="<<contrib_Qoverp+contrib_dxy+contrib_dz+contrib_theta+contrib_phi<<"\n";
	  
	  h_pullQoverp[w]->Fill(qoverpPull);
	  h_pullTheta[w]->Fill(thetaPull);
	  h_pullPhi[w]->Fill(phiPull);
	  h_pullDxy[w]->Fill(dxyPull);
	  h_pullDz[w]->Fill(dzPull);

	  double ptres=ptSeed-sqrt(assocTrack->momentum().perp2()); 
	  //double etares=etaSeed-assocTrack->momentum().pseudoRapidity();
	  double etares=etaSeed-assocTrack->momentum().Eta();

	  h_pt[w]->Fill(ptres/ptErrorSeed);
	  h_eta[w]->Fill(etares);

	  //#hit vs eta: fill 2D histos
	  nhits_vs_eta[w]->Fill(getEta(etaSeed),numberOfHitsSeed);

	} catch (cms::Exception e){
	  LogTrace("SeedValidator") << "exception found: " << e.what() << "\n";
	}
      }
      if (at!=0) h_tracks[w]->Fill(at);
      h_fakes[w]->Fill(rT-at);
      edm::LogVerbatim("SeedValidator") << "Total Simulated: " << st << "\n"
					 << "Total Associated (simToReco): " << ats << "\n"
					 << "Total Reconstructed: " << rT << "\n"
					 << "Total Associated (recoToSim): " << at << "\n"
					 << "Total Fakes: " << rT-at << "\n";
      nrec_vs_nsim[w]->Fill(rT,st);
      w++;
    }
  }
}

void TrackerSeedValidator::endRun(edm::Run const&, edm::EventSetup const&) {
  LogTrace("SeedValidator") << "TrackerSeedValidator::endRun()";
  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){

      doProfileX(nhits_vs_eta[w],h_hits_eta[w]);    

      //effic&fake
      fillPlotFromVectors(h_effic[w],totASSeta[w],totSIMeta[w],"effic");
      fillPlotFromVectors(h_fakerate[w],totASS2eta[w],totRECeta[w],"fakerate");
      fillPlotFromVectors(h_efficPt[w],totASSpT[w],totSIMpT[w],"effic");
      fillPlotFromVectors(h_fakeratePt[w],totASS2pT[w],totRECpT[w],"fakerate");
      fillPlotFromVectors(h_effic_vs_hit[w],totASS_hit[w],totSIM_hit[w],"effic");
      fillPlotFromVectors(h_fake_vs_hit[w],totASS2_hit[w],totREC_hit[w],"fakerate");

      fillPlotFromVector(h_recoeta[w],totRECeta[w]);
      fillPlotFromVector(h_simuleta[w],totSIMeta[w]);
      fillPlotFromVector(h_assoceta[w],totASSeta[w]);
      fillPlotFromVector(h_assoc2eta[w],totASS2eta[w]);

      fillPlotFromVector(h_recopT[w],totRECpT[w]);
      fillPlotFromVector(h_simulpT[w],totSIMpT[w]);
      fillPlotFromVector(h_assocpT[w],totASSpT[w]);
      fillPlotFromVector(h_assoc2pT[w],totASS2pT[w]);

      w++;
    }
  }
  if ( out.size() != 0 && dbe_ ) dbe_->save(out);
}
*/



