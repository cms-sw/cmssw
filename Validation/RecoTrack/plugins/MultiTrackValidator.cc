#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "Validation/Tools/interface/FitSlicesYTool.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"

#include "TMath.h"
#include <TF1.h>

using namespace std;
using namespace edm;

void MultiTrackValidator::beginJob( const EventSetup & setup) {

  dbe_->showDirStructure();

  int j=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){

      dbe_->cd();
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
      h_tracks.push_back( dbe_->book1D("tracks","number of reconstructed tracks",20,-0.5,19.5) );
      h_fakes.push_back( dbe_->book1D("fakes","number of fake reco tracks",20,-0.5,19.5) );
      h_charge.push_back( dbe_->book1D("charge","charge",3,-1.5,1.5) );
      h_hits.push_back( dbe_->book1D("hits", "number of hits per track", 35, -0.5, 34.5 ) );
      h_losthits.push_back( dbe_->book1D("losthits", "number of lost hits per track", 35, -0.5, 34.5 ) );
      h_nchi2.push_back( dbe_->book1D("chi2", "normalized #chi^{2}", 200, 0, 20 ) );
      h_nchi2_prob.push_back( dbe_->book1D("chi2_prob", "normalized #chi^{2} probability",100,0,1));

      h_effic.push_back( dbe_->book1D("effic","efficiency vs #eta",nint,min,max) );
      h_efficPt.push_back( dbe_->book1D("efficPt","efficiency vs pT",nintpT,minpT,maxpT) );
      h_fakerate.push_back( dbe_->book1D("fakerate","fake rate vs #eta",nint,min,max) );
      h_fakeratePt.push_back( dbe_->book1D("fakeratePt","fake rate vs pT",nintpT,minpT,maxpT) );
      h_effic_vs_hit.push_back( dbe_->book1D("effic_vs_hit","effic vs hit",nintHit,minHit,maxHit) );
      h_fake_vs_hit.push_back( dbe_->book1D("fakerate_vs_hit","fake rate vs hit",nintHit,minHit,maxHit) );
 
      h_recoeta.push_back( dbe_->book1D("num_reco_eta","N of reco track vs eta",nint,min,max) );
      h_assoceta.push_back( dbe_->book1D("num_assoc(simToReco)_eta","N of associated tracks (simToReco) vs eta",nint,min,max) );
      h_assoc2eta.push_back( dbe_->book1D("num_assoc(recoToSim)_eta","N of associated (recoToSim) tracks vs eta",nint,min,max) );
      h_simuleta.push_back( dbe_->book1D("num_simul_eta","N of simulated tracks vs eta",nint,min,max) );
      h_recopT.push_back( dbe_->book1D("num_reco_pT","N of reco track vs pT",nintpT,minpT,maxpT) );
      h_assocpT.push_back( dbe_->book1D("num_assoc(simToReco)_pT","N of associated tracks (simToReco) vs pT",nintpT,minpT,maxpT) );
      h_assoc2pT.push_back( dbe_->book1D("num_assoc(recoToSim)_pT","N of associated (recoToSim) tracks vs pT",nintpT,minpT,maxpT) );
      h_simulpT.push_back( dbe_->book1D("num_simul_pT","N of simulated tracks vs pT",nintpT,minpT,maxpT) );
      
      h_eta.push_back( dbe_->book1D("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
      h_pt.push_back( dbe_->book1D("pullPt", "pull of p_{t}", 100, -10, 10 ) );
      h_pullTheta.push_back( dbe_->book1D("pullTheta","pull of #theta parameter",250,-25,25) );
      h_pullPhi.push_back( dbe_->book1D("pullPhi","pull of #phi parameter",250,-25,25) );
      h_pullDxy.push_back( dbe_->book1D("pullDxy","pull of dxy parameter",250,-25,25) );
      h_pullDz.push_back( dbe_->book1D("pullDz","pull of dz parameter",250,-25,25) );
      h_pullQoverp.push_back( dbe_->book1D("pullQoverp","pull of qoverp parameter",250,-25,25) );
      
      if (associators[ww]=="TrackAssociatorByChi2"){
	h_assochi2.push_back( dbe_->book1D("assocChi2","track association #chi^{2}",1000000,0,100000) );
	h_assochi2_prob.push_back(dbe_->book1D("assocChi2_prob","probability of association #chi^{2}",100,0,1));
      } else if (associators[ww]=="TrackAssociatorByHits"){
	h_assocFraction.push_back( dbe_->book1D("assocFraction","fraction of shared hits",200,0,2) );
	h_assocSharedHit.push_back(dbe_->book1D("assocSharedHit","number of shared hits",20,0,20));
      }

      chi2_vs_nhits.push_back( dbe_->book2D("chi2_vs_nhits","#chi^{2} vs nhits",25,0,25,100,0,10) );
      etares_vs_eta.push_back( dbe_->book2D("etares_vs_eta","etaresidue vs eta",nint,min,max,200,-0.1,0.1) );
      nrec_vs_nsim.push_back( dbe_->book2D("nrec_vs_nsim","nrec vs nsim",20,-0.5,19.5,20,-0.5,19.5) );

      chi2_vs_eta.push_back( dbe_->book2D("chi2_vs_eta","chi2_vs_eta",nint,min,max, 200, 0, 20 ));
      h_chi2meanh.push_back( dbe_->book1D("chi2mean","mean #chi^{2} vs #eta",nint,min,max) );

      nhits_vs_eta.push_back( dbe_->book2D("nhits_vs_eta","nhits vs eta",nint,min,max,25,0,25) );
      h_hits_eta.push_back( dbe_->book1D("hits_eta","mean #hits vs eta",nint,min,max) );

      nlosthits_vs_eta.push_back( dbe_->book2D("nlosthits_vs_eta","nlosthits vs eta",nint,min,max,25,0,25) );
      h_losthits_eta.push_back( dbe_->book1D("losthits_eta","losthits_eta",nint,min,max) );

      //resolution of track parameters
      //                       dPt/Pt    cotTheta        Phi            TIP            LIP
      // log10(pt)<0.5        100,0.1    240,0.08     100,0.015      100,0.1000    150,0.3000
      // 0.5<log10(pt)<1.5    100,0.1    120,0.01     100,0.003      100,0.0100    150,0.0500
      // >1.5                 100,0.3    100,0.005    100,0.0008     100,0.0060    120,0.0300

      ptres_vs_eta.push_back(dbe_->book2D("ptres_vs_eta","ptres_vs_eta",nint,min,max, 100, -0.1, 0.1));
      h_ptrmsh.push_back( dbe_->book1D("sigmapt","#sigma(#deltap_{t}/p_{t}) vs #eta",nint,min,max) );

      ptres_vs_pt.push_back(dbe_->book2D("ptres_vs_pt","ptres_vs_pt",nintpT,minpT,maxpT, 100, -0.1, 0.1));
      h_ptrmshPt.push_back( dbe_->book1D("sigmaptPt","#sigma(#deltap_{t}/p_{t}) vs pT",nintpT,minpT,maxpT) );

      cotThetares_vs_eta.push_back(dbe_->book2D("cotThetares_vs_eta","cotThetares_vs_eta",nint,min,max, 120, -0.01, 0.01));
      h_cotThetarmsh.push_back( dbe_->book1D("sigmacotTheta","#sigma(#deltacot(#theta)) vs #eta",nint,min,max) );

      cotThetares_vs_pt.push_back(dbe_->book2D("cotThetares_vs_pt","cotThetares_vs_pt",nintpT,minpT,maxpT, 120, -0.01, 0.01));
      h_cotThetarmshPt.push_back( dbe_->book1D("sigmacotThetaPt","#sigma(#deltacot(#theta)) vs pT",nintpT,minpT,maxpT) );

      phires_vs_eta.push_back(dbe_->book2D("phires_vs_eta","phires_vs_eta",nint,min,max, 100, -0.003, 0.003));
      h_phirmsh.push_back( dbe_->book1D("sigmaphi","#sigma(#delta#phi) vs #eta",nint,min,max) );

      phires_vs_pt.push_back(dbe_->book2D("phires_vs_pt","phires_vs_pt",nintpT,minpT,maxpT, 100, -0.003, 0.003));
      h_phirmshPt.push_back( dbe_->book1D("sigmaphiPt","#sigma(#delta#phi) vs pT",nintpT,minpT,maxpT) );

      dxyres_vs_eta.push_back(dbe_->book2D("dxyres_vs_eta","dxyres_vs_eta",nint,min,max, 100, -0.01, 0.01));
      h_dxyrmsh.push_back( dbe_->book1D("sigmadxy","#sigma(#deltadxy) vs #eta",nint,min,max) );

      dxyres_vs_pt.push_back( dbe_->book2D("dxyres_vs_pt","dxyres_vs_pt",nintpT,minpT,maxpT, 100, -0.01, 0.01));
      h_dxyrmshPt.push_back( dbe_->book1D("sigmadxyPt","#sigmadxy vs pT",nintpT,minpT,maxpT) );

      dzres_vs_eta.push_back(dbe_->book2D("dzres_vs_eta","dzres_vs_eta",nint,min,max, 150, -0.05, 0.05));
      h_dzrmsh.push_back( dbe_->book1D("sigmadz","#sigma(#deltadz) vs #eta",nint,min,max) );

      dzres_vs_pt.push_back(dbe_->book2D("dzres_vs_pt","dzres_vs_pt",nintpT,minpT,maxpT, 150, -0.05, 0.05));
      h_dzrmshPt.push_back( dbe_->book1D("sigmadzPt","#sigma(#deltadz vs pT",nintpT,minpT,maxpT) );

      //pulls of track params vs eta: to be used with fitslicesytool
      dxypull_vs_eta.push_back(dbe_->book2D("dxypull_vs_eta","dxypull_vs_eta",nint,min,max,100,-10,10));
      ptpull_vs_eta.push_back(dbe_->book2D("ptpull_vs_eta","ptpull_vs_eta",nint,min,max,100,-10,10)); 
      dzpull_vs_eta.push_back(dbe_->book2D("dzpull_vs_eta","dzpull_vs_eta",nint,min,max,100,-10,10)); 
      phipull_vs_eta.push_back(dbe_->book2D("phipull_vs_eta","phipull_vs_eta",nint,min,max,100,-10,10)); 
      thetapull_vs_eta.push_back(dbe_->book2D("thetapull_vs_eta","thetapull_vs_eta",nint,min,max,100,-10,10));
      h_dxypulleta.push_back( dbe_->book1D("h_dxypulleta","#sigma of dxy pull vs #eta",nint,min,max) ); 
      h_ptpulleta.push_back( dbe_->book1D("h_ptpulleta","#sigma of p_{t} pull vs #eta",nint,min,max) ); 
      h_dzpulleta.push_back( dbe_->book1D("h_dzpulleta","#sigma of dz pull vs #eta",nint,min,max) ); 
      h_phipulleta.push_back( dbe_->book1D("h_phipulleta","#sigma of #phi pull vs #eta",nint,min,max) ); 
      h_thetapulleta.push_back( dbe_->book1D("h_thetapulleta","#sigma of #theta pull vs #eta",nint,min,max) );
      h_ptshifteta.push_back( dbe_->book1D("h_ptshifteta","<#deltapT/pT>[%] vs #eta",nint,min,max) ); 

      j++;
    }
  }
  edm::ESHandle<TrackAssociatorBase> theAssociator;
  for (unsigned int w=0;w<associators.size();w++) {
    setup.get<TrackAssociatorRecord>().get(associators[w],theAssociator);
    associator.push_back( theAssociator.product() );
  }
}

void MultiTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){
  using namespace reco;

  edm::LogInfo("TrackValidator") << "\n====================================================" << "\n"
				 << "Analyzing new event" << "\n"
				 << "====================================================\n" << "\n";
  
  edm::Handle<TrackingParticleCollection>  TPCollectionHeff ;
  event.getByLabel(label_tp_effic,TPCollectionHeff);
  const TrackingParticleCollection tPCeff = *(TPCollectionHeff.product());
  
  edm::Handle<TrackingParticleCollection>  TPCollectionHfake ;
  event.getByLabel(label_tp_fake,TPCollectionHfake);
  const TrackingParticleCollection tPCfake = *(TPCollectionHfake.product());

  //if (tPCeff.size()==0) {edm::LogInfo("TrackValidator") << "TP Collection for efficiency studies has size = 0! Skipping Event." ; return;}
  //if (tPCfake.size()==0) {edm::LogInfo("TrackValidator") << "TP Collection for fake rate studies has size = 0! Skipping Event." ; return;}

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
      edm::Handle<View<Track> >  trackCollection;
      event.getByLabel(label[www], trackCollection);
      //if (trackCollection->size()==0) {
      //edm::LogInfo("TrackValidator") << "TrackCollection size = 0!" ; 
      //continue;
      //}
      
      //associate tracks
      LogTrace("TrackValidator") << "Calling associateRecoToSim method" << "\n";
      reco::RecoToSimCollection recSimColl=associator[ww]->associateRecoToSim(trackCollection,
									      TPCollectionHfake,
									      &event);
      LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
      reco::SimToRecoCollection simRecColl=associator[ww]->associateSimToReco(trackCollection,
									      TPCollectionHeff, 
									      &event);

      //
      //fill simulation histograms
      //compute number of tracks per eta interval
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

	std::vector<std::pair<RefToBase<Track>, double> > rt;
	if(simRecColl.find(tp) != simRecColl.end()){
	  rt = (std::vector<std::pair<RefToBase<Track>, double> >) simRecColl[tp];
	  if (rt.size()!=0) {
	    ats++;
	    edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st 
					       << " with pt=" << sqrt(tp->momentum().perp2()) 
					       << " associated with quality:" << rt.begin()->second <<"\n";
	  }
	}else{
	  edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st
					     << " with pt=" << sqrt(tp->momentum().perp2())
					     << " NOT associated to any reco::Track" << "\n";
	}

	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (getEta(tp->momentum().eta())>etaintervals[w][f]&&
	      getEta(tp->momentum().eta())<etaintervals[w][f+1]) {
	    totSIMeta[w][f]++;
	    if (rt.size()!=0) {
	      totASSeta[w][f]++;
	    }
	  }
	} // END for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	
	for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
          if (getPt(sqrt(tp->momentum().perp2()))>pTintervals[w][f]&&
              getPt(sqrt(tp->momentum().perp2()))<pTintervals[w][f+1]) {
            totSIMpT[w][f]++;
	    if (rt.size()!=0) {
	      totASSpT[w][f]++;
	    }
	  }
	} // END for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
	totSIM_hit[w][std::min(tp->matchedHit(),nintHit-1)]++;
	if (rt.size()!=0) totASS_hit[w][std::min(tp->matchedHit(),nintHit-1)]++;
      }
      if (st!=0) h_tracksSIM[w]->Fill(st);
      

      //
      //fill reconstructed track histograms
      // 
      edm::LogVerbatim("TrackValidator") << "\n# of reco::Tracks with "
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()
					 << ": " << trackCollection->size() << "\n";
      int at=0;
      int rT=0;
      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){
	RefToBase<Track> track(trackCollection, i);
	rT++;

	std::vector<std::pair<TrackingParticleRef, double> > tp;
	if(recSimColl.find(track) != recSimColl.end()){
	  tp = recSimColl[track];
	  if (tp.size()!=0) {
	    at++;
	    edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt() 
					       << " associated with quality:" << tp.begin()->second <<"\n";
	  }
	} else {
	  edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
					     << " NOT associated to any TrackingParticle" << "\n";		  
	}
	
	//Compute fake rate vs eta
	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (getEta(track->momentum().eta())>etaintervals[w][f]&&
	      getEta(track->momentum().eta())<etaintervals[w][f+1]) {
	    totRECeta[w][f]++; 
	    if (tp.size()!=0) {
	      totASS2eta[w][f]++;
	    }		
	  }
	}
	
	for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
	  if (getPt(sqrt(track->momentum().perp2()))>pTintervals[w][f]&&
	      getPt(sqrt(track->momentum().perp2()))<pTintervals[w][f+1]) {
	    totRECpT[w][f]++; 
	    if (tp.size()!=0) {
	      totASS2pT[w][f]++;
	    }	      
	  }
	}
	int tmp = std::min((int)track->found(),int(maxHit-1));
 	totREC_hit[w][tmp]++;
	if (tp.size()!=0) totASS2_hit[w][tmp]++;

	//Fill other histos
 	try{
	  if (tp.size()==0) continue;
	
	  TrackingParticleRef tpr = tp.begin()->first;
	  const SimTrack * assocTrack = &(*tpr->g4Track_begin());
	
	  if (associators[ww]=="TrackAssociatorByChi2"){
	    //association chi2
	    double assocChi2 = -tp.begin()->second;//in association map is stored -chi2
	    h_assochi2[www]->Fill(assocChi2);
	    h_assochi2_prob[www]->Fill(TMath::Prob((assocChi2)*5,5));
	  }
	  else if (associators[ww]=="TrackAssociatorByHits"){
	    double fraction = tp.begin()->second;
	    h_assocFraction[www]->Fill(fraction);
	    h_assocSharedHit[www]->Fill(fraction*track->numberOfValidHits());
	  }
    
	  //nchi2 and hits global distributions
	  h_nchi2[w]->Fill(track->normalizedChi2());
	  h_nchi2_prob[w]->Fill(TMath::Prob(track->chi2(),(int)track->ndof()));
	  h_hits[w]->Fill(track->numberOfValidHits());
	  h_losthits[w]->Fill(track->numberOfLostHits());
	  chi2_vs_nhits[w]->Fill(track->numberOfValidHits(),track->normalizedChi2());
	  h_charge[w]->Fill( track->charge() );
	
	  //compute tracking particle parameters at point of closest approach to the beamline
	  edm::ESHandle<MagneticField> theMF;
	  setup.get<IdealMagneticFieldRecord>().get(theMF);
	  FreeTrajectoryState 
	    ftsAtProduction(GlobalPoint(tpr->vertex().x(),tpr->vertex().y(),tpr->vertex().z()),
			    GlobalVector(assocTrack->momentum().x(),assocTrack->momentum().y(),assocTrack->momentum().z()),
			    TrackCharge(tpr->charge()),
			    theMF.product());
	  TrajectoryStateClosestToBeamLineBuilder tscblBuilder;
	  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(ftsAtProduction,bs);//as in TrackProducerAlgorithm
	  GlobalPoint v1 = tsAtClosestApproach.trackStateAtPCA().position();
	  GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();
	  GlobalPoint v(v1.x()-bs.x0(),v1.y()-bs.y0(),v1.z()-bs.z0());

	  double qoverpSim = tsAtClosestApproach.trackStateAtPCA().charge()/p.mag();
	  double lambdaSim = M_PI/2-p.theta();
	  double phiSim    = p.phi();
	  double dxySim    = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
	  double dzSim     = v.z() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.perp();

	  TrackBase::ParameterVector rParameters = track->parameters();	  
	  double qoverpRec = rParameters[0];
	  double lambdaRec = rParameters[1];
	  double phiRec    = rParameters[2];
	  double dxyRec    = track->dxy(bs.position());
	  double dzRec     = track->dz(bs.position());

	  // eta residue; pt, k, theta, phi, dxy, dz pulls
	  double qoverpPull=(qoverpRec-qoverpSim)/track->qoverpError();
	  double thetaPull=(lambdaRec-lambdaSim)/track->thetaError();
	  double phiPull=(phiRec-phiSim)/track->phiError();
	  double dxyPull=(dxyRec-dxySim)/track->dxyError();
	  double dzPull=(dzRec-dzSim)/track->dzError();

	  double contrib_Qoverp = ((qoverpRec-qoverpSim)/track->qoverpError())*
	    ((qoverpRec-qoverpSim)/track->qoverpError())/5;
	  double contrib_dxy = ((dxyRec-dxySim)/track->dxyError())*((dxyRec-dxySim)/track->dxyError())/5;
	  double contrib_dz = ((dzRec-dzSim)/track->dzError())*((dzRec-dzSim)/track->dzError())/5;
	  double contrib_theta = ((lambdaRec-lambdaSim)/track->thetaError())*
	    ((lambdaRec-lambdaSim)/track->thetaError())/5;
	  double contrib_phi = ((phiRec-phiSim)/track->phiError())*
	    ((phiRec-phiSim)/track->phiError())/5;
	  LogTrace("TrackValidatorTEST") << "assocChi2=" << tp.begin()->second << "\n"
					 << "" <<  "\n"
					 << "ptREC=" << track->pt() << "\n"
					 << "etaREC=" << track->eta() << "\n"
					 << "qoverpREC=" << qoverpRec << "\n"
					 << "dxyREC=" << dxyRec << "\n"
					 << "dzREC=" << dzRec << "\n"
					 << "thetaREC=" << track->theta() << "\n"
					 << "phiREC=" << phiRec << "\n"
					 << "" <<  "\n"
					 << "qoverpError()=" << track->qoverpError() << "\n"
					 << "dxyError()=" << track->dxyError() << "\n"
					 << "dzError()=" << track->dzError() << "\n"
					 << "thetaError()=" << track->thetaError() << "\n"
					 << "phiError()=" << track->phiError() << "\n"
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

	  double ptres=track->pt()-sqrt(assocTrack->momentum().perp2()); 
	  double etares=track->eta()-assocTrack->momentum().Eta();

	  double ptError =  track->ptError();
	  h_pt[w]->Fill(ptres/ptError);
	  h_eta[w]->Fill(etares);
	  etares_vs_eta[w]->Fill(getEta(track->eta()),etares);

	  //chi2 and #hit vs eta: fill 2D histos
	  chi2_vs_eta[w]->Fill(getEta(track->eta()),track->normalizedChi2());
	  nhits_vs_eta[w]->Fill(getEta(track->eta()),track->numberOfValidHits());
	  nlosthits_vs_eta[w]->Fill(getEta(track->eta()),track->numberOfLostHits());

	  //resolution of track params: fill 2D histos
	  dxyres_vs_eta[w]->Fill(getEta(track->eta()),dxyRec-dxySim);
	  ptres_vs_eta[w]->Fill(getEta(track->eta()),(track->pt()-sqrt(assocTrack->momentum().perp2()))/track->pt());
	  dzres_vs_eta[w]->Fill(getEta(track->eta()),dzRec-dzSim);
	  phires_vs_eta[w]->Fill(getEta(track->eta()),phiRec-phiSim);
	  cotThetares_vs_eta[w]->Fill(getEta(track->eta()),1/tan(1.570796326794896558-lambdaRec)-1/tan(1.570796326794896558-lambdaSim));         

	  //same as before but vs pT
	  dxyres_vs_pt[w]->Fill(getPt(track->pt()),dxyRec-dxySim);
	  ptres_vs_pt[w]->Fill(getPt(track->pt()),(track->pt()-sqrt(assocTrack->momentum().perp2()))/track->pt());
	  dzres_vs_pt[w]->Fill(getPt(track->pt()),dzRec-dzSim);
	  phires_vs_pt[w]->Fill(getPt(track->pt()),phiRec-phiSim);
	  cotThetares_vs_pt[w]->Fill(getPt(track->pt()),1/tan(1.570796326794896558-lambdaRec)-1/tan(1.570796326794896558-lambdaSim));  	 
  	 
	  //pulls of track params vs eta: fill 2D histos
	  dxypull_vs_eta[w]->Fill(getEta(track->eta()),dxyPull);
	  ptpull_vs_eta[w]->Fill(getEta(track->eta()),ptres/ptError);
	  dzpull_vs_eta[w]->Fill(getEta(track->eta()),dzPull);
	  phipull_vs_eta[w]->Fill(getEta(track->eta()),phiPull);
	  thetapull_vs_eta[w]->Fill(getEta(track->eta()),thetaPull);

	} catch (cms::Exception e){
	  LogTrace("TrackValidator") << "exception found: " << e.what() << "\n";
	}
      }
      if (at!=0) h_tracks[w]->Fill(at);
      h_fakes[w]->Fill(rT-at);
      edm::LogVerbatim("TrackValidator") << "Total Simulated: " << st << "\n"
					 << "Total Associated (simToReco): " << ats << "\n"
					 << "Total Reconstructed: " << rT << "\n"
					 << "Total Associated (recoToSim): " << at << "\n"
					 << "Total Fakes: " << rT-at << "\n";
      nrec_vs_nsim[w]->Fill(rT,st);
      w++;
    }
  }
}

void MultiTrackValidator::endJob() {
  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){

      //resolution of track params: get sigma from 2D histos
      FitSlicesYTool fsyt_dxy(dxyres_vs_eta[w]);
      fsyt_dxy.getFittedSigmaWithError(h_dxyrmsh[w]);
      FitSlicesYTool fsyt_dxyPt(dxyres_vs_pt[w]);
      fsyt_dxyPt.getFittedSigmaWithError(h_dxyrmshPt[w]);
      FitSlicesYTool fsyt_pt(ptres_vs_eta[w]);
      fsyt_pt.getFittedSigmaWithError(h_ptrmsh[w]);
      fsyt_pt.getFittedMeanWithError(h_ptshifteta[w]);      
      FitSlicesYTool fsyt_ptPt(ptres_vs_pt[w]);
      fsyt_ptPt.getFittedSigmaWithError(h_ptrmshPt[w]);
      FitSlicesYTool fsyt_dz(dzres_vs_eta[w]);
      fsyt_dz.getFittedSigmaWithError(h_dzrmsh[w]);
      FitSlicesYTool fsyt_dzPt(dzres_vs_pt[w]);
      fsyt_dzPt.getFittedSigmaWithError(h_dzrmshPt[w]);
      FitSlicesYTool fsyt_phi(phires_vs_eta[w]);
      fsyt_phi.getFittedSigmaWithError(h_phirmsh[w]);
      FitSlicesYTool fsyt_phiPt(phires_vs_pt[w]);
      fsyt_phiPt.getFittedSigmaWithError(h_phirmshPt[w]);
      FitSlicesYTool fsyt_cotTheta(cotThetares_vs_eta[w]);
      fsyt_cotTheta.getFittedSigmaWithError(h_cotThetarmsh[w]);
      FitSlicesYTool fsyt_cotThetaPt(cotThetares_vs_pt[w]);
      fsyt_cotThetaPt.getFittedSigmaWithError(h_cotThetarmshPt[w]);

      //chi2 and #hit vs eta: get mean from 2D histos
      doProfileX(chi2_vs_eta[w],h_chi2meanh[w]);
      doProfileX(nhits_vs_eta[w],h_hits_eta[w]);    
   
      //pulls of track params vs eta: get sigma from 2D histos
      FitSlicesYTool fsyt_dxyp(dxypull_vs_eta[w]);
      fsyt_dxyp.getFittedSigmaWithError(h_dxypulleta[w]);
      FitSlicesYTool fsyt_ptp(ptpull_vs_eta[w]);
      fsyt_ptp.getFittedSigmaWithError(h_ptpulleta[w]);
      FitSlicesYTool fsyt_dzp(dzpull_vs_eta[w]);
      fsyt_dzp.getFittedSigmaWithError(h_dzpulleta[w]);
      FitSlicesYTool fsyt_phip(phipull_vs_eta[w]);
      fsyt_phip.getFittedSigmaWithError(h_phipulleta[w]);
      FitSlicesYTool fsyt_thetap(thetapull_vs_eta[w]);
      fsyt_thetap.getFittedSigmaWithError(h_thetapulleta[w]);
      
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

