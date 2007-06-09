#include "Validation/RecoTrack/interface/MultiTrackValidator.h"
#include "Validation/RecoTrack/interface/FitSlicesYTool.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "Math/ProbFuncMathMore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TF1.h>

using namespace std;
using namespace ROOT::Math;
using namespace edm;

void MultiTrackValidator::beginJob( const EventSetup & setup) {

  dbe_->showDirStructure();

  int j=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){

      dbe_->cd();
      InputTag algo = label[www];
      string dirName="Track/";
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

      vector<double> etaintervalsv;
      vector<double> pTintervalsv;
      vector<int>    totSIMveta,totASSveta,totASS2veta,totRECveta;
      vector<int>    totSIMvpT,totASSvpT,totASS2vpT,totRECvpT;
  
      double step=(max-min)/nint;
      ostringstream title,name;
      etaintervalsv.push_back(0);
      for (int k=1;k<nint+1;k++) {
	double d=k*step;
	etaintervalsv.push_back(d);
	totSIMveta.push_back(0);
	totASSveta.push_back(0);
	totASS2veta.push_back(0);
	totRECveta.push_back(0);
      }

      etaintervals.push_back(etaintervalsv);
      totSIMeta.push_back(totSIMveta);
      totASSeta.push_back(totASSveta);
      totASS2eta.push_back(totASS2veta);
      totRECeta.push_back(totRECveta);

      double steppT = (maxpT-minpT)/nintpT;
      pTintervalsv.push_back(0);
      for (int k=1;k<nintpT+1;k++) {
        double d=k*steppT;
        pTintervalsv.push_back(d);
        totSIMvpT.push_back(0);
        totASSvpT.push_back(0);
        totASS2vpT.push_back(0);
        totRECvpT.push_back(0);
      }
      pTintervals.push_back(pTintervalsv);
      totSIMpT.push_back(totSIMvpT);
      totASSpT.push_back(totASSvpT);
      totASS2pT.push_back(totASS2vpT);
      totRECpT.push_back(totRECvpT);

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
      h_hits.push_back( dbe_->book1D("hits", "number of hits per track", 30, -0.5, 29.5 ) );
      h_nchi2.push_back( dbe_->book1D("chi2", "normalized #chi^{2}", 200, 0, 20 ) );
      h_nchi2_prob.push_back( dbe_->book1D("chi2_prob", "normalized #chi^{2} probability",100,0,1));

      h_effic.push_back( dbe_->book1D("effic","efficiency vs #eta",nint,min,max) );
      h_efficPt.push_back( dbe_->book1D("efficPt","efficiency vs pT",nintpT,minpT,maxpT) );
      h_fakerate.push_back( dbe_->book1D("fakerate","fake rate vs #eta",nint,min,max) );
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
      h_pullPhi0.push_back( dbe_->book1D("pullPhi0","pull of #phi0 parameter",250,-25,25) );
      h_pullD0.push_back( dbe_->book1D("pullD0","pull of d0 parameter",250,-25,25) );
      h_pullDz.push_back( dbe_->book1D("pullDz","pull of dz parameter",250,-25,25) );
      h_pullQoverp.push_back( dbe_->book1D("pullQoverp","pull of qoverp parameter",250,-25,25) );
      
      if (associators[ww]=="TrackAssociatorByChi2"){
	h_assochi2.push_back( dbe_->book1D("assocChi2","track association #chi^{2}",1000000,0,100000) );
	h_assochi2_prob.push_back(dbe_->book1D("assocChi2_prob","probability of association #chi^{2}",100,0,1));
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
      h_ptrmshPt.push_back( dbe_->book1D("sigmaptPt","#sigma(#deltap_{t}/p_{t}) vs #pt",nintpT,minpT,maxpT) );

      cotThetares_vs_eta.push_back(dbe_->book2D("cotThetares_vs_eta","cotThetares_vs_eta",nint,min,max, 120, -0.01, 0.01));
      h_cotThetarmsh.push_back( dbe_->book1D("sigmacotTheta","#sigma(#deltacot(#theta)) vs #eta",nint,min,max) );

      cotThetares_vs_pt.push_back(dbe_->book2D("cotThetares_vs_pt","cotThetares_vs_pt",nintpT,minpT,maxpT, 120, -0.01, 0.01));
      h_cotThetarmshPt.push_back( dbe_->book1D("sigmacotThetaPt","#sigma(#deltacot(#theta)) vs #pt",nintpT,minpT,maxpT) );

      phires_vs_eta.push_back(dbe_->book2D("phires_vs_eta","phires_vs_eta",nint,min,max, 100, -0.003, 0.003));
      h_phirmsh.push_back( dbe_->book1D("sigmaphi","#sigma(#delta#phi) vs #eta",nint,min,max) );

      phires_vs_pt.push_back(dbe_->book2D("phires_vs_pt","phires_vs_pt",nintpT,minpT,maxpT, 100, -0.003, 0.003));
      h_phirmshPt.push_back( dbe_->book1D("sigmaphiPt","#sigma(#delta#phi) vs #pt",nintpT,minpT,maxpT) );

      d0res_vs_eta.push_back(dbe_->book2D("d0res_vs_eta","d0res_vs_eta",nint,min,max, 100, -0.01, 0.01));
      h_d0rmsh.push_back( dbe_->book1D("sigmad0","#sigma(#deltad_{0}) vs #eta",nint,min,max) );

      d0res_vs_pt.push_back( dbe_->book2D("d0res_vs_pt","d0res_vs_pt",nintpT,minpT,maxpT, 100, -0.01, 0.01));
      h_d0rmshPt.push_back( dbe_->book1D("sigmad0Pt","#sigmad0 vs pT",nintpT,minpT,maxpT) );

      z0res_vs_eta.push_back(dbe_->book2D("z0res_vs_eta","z0res_vs_eta",nint,min,max, 150, -0.05, 0.05));
      h_z0rmsh.push_back( dbe_->book1D("sigmaz0","#sigma(#deltaz_{0}) vs #eta",nint,min,max) );

      z0res_vs_pt.push_back(dbe_->book2D("z0res_vs_pt","z0res_vs_pt",nintpT,minpT,maxpT, 150, -0.05, 0.05));
      h_z0rmshPt.push_back( dbe_->book1D("sigmaz0Pt","#sigma(#deltaz_{0}) vs #pt",nintpT,minpT,maxpT) );

      //pulls of track params vs eta: to be used with fitslicesytool
      d0pull_vs_eta.push_back(dbe_->book2D("d0pull_vs_eta","d0pull_vs_eta",nint,min,max,100,-10,10));
      ptpull_vs_eta.push_back(dbe_->book2D("ptpull_vs_eta","ptpull_vs_eta",nint,min,max,100,-10,10)); 
      z0pull_vs_eta.push_back(dbe_->book2D("z0pull_vs_eta","z0pull_vs_eta",nint,min,max,100,-10,10)); 
      phipull_vs_eta.push_back(dbe_->book2D("phipull_vs_eta","phipull_vs_eta",nint,min,max,100,-10,10)); 
      thetapull_vs_eta.push_back(dbe_->book2D("thetapull_vs_eta","thetapull_vs_eta",nint,min,max,100,-10,10));
      h_d0pulleta.push_back( dbe_->book1D("h_d0pulleta","#sigma of d0 pull vs #eta",nint,min,max) ); 
      h_ptpulleta.push_back( dbe_->book1D("h_ptpulleta","#sigma of p_{t} pull vs #eta",nint,min,max) ); 
      h_z0pulleta.push_back( dbe_->book1D("h_z0pulleta","#sigma of z0 pull vs #eta",nint,min,max) ); 
      h_phipulleta.push_back( dbe_->book1D("h_phipulleta","#sigma of #phi pull vs #eta",nint,min,max) ); 
      h_thetapulleta.push_back( dbe_->book1D("h_thetapulleta","#sigma of #theta pull vs #eta",nint,min,max) );

      j++;
    }
  }
  edm::ESHandle<TrackAssociatorBase> theAssociator;
  for (unsigned int w=0;w<associators.size();w++) {
    setup.get<TrackAssociatorRecord>().get(associators[w],theAssociator);
    associator.push_back( (const TrackAssociatorBase *) theAssociator.product() );
  }
  
  edm::ESHandle<TrackAssociatorBase> theAssociatorForParamAtPca;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2",theAssociatorForParamAtPca);
  associatorForParamAtPca = (const TrackAssociatorByChi2 *) theAssociatorForParamAtPca.product();
}

void MultiTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){

  edm::LogInfo("TrackValidator") << "\n====================================================" << "\n"
			       << "Analyzing new event" << "\n"
			       << "====================================================\n" << "\n";

  edm::Handle<TrackingParticleCollection>  TPCollectionHeff ;
  event.getByLabel(label_tp_effic,TPCollectionHeff);
  const TrackingParticleCollection tPCeff = *(TPCollectionHeff.product());
  
  edm::Handle<TrackingParticleCollection>  TPCollectionHfake ;
  event.getByLabel(label_tp_fake,TPCollectionHfake);
  const TrackingParticleCollection tPCfake = *(TPCollectionHfake.product());
  
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
      edm::Handle<reco::TrackCollection> trackCollection;
      event.getByLabel(label[www], trackCollection);
      const reco::TrackCollection tC = *(trackCollection.product());
      
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
      edm::LogVerbatim("TrackValidator") << "\n# of TrackingParticles (before cuts): " << tPCeff.size() << "\n";
      int ats = 0;
      int st=0;
      for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
	TrackingParticleRef tp(TPCollectionHeff, i);
	//if (!selectTPs4Efficiency( *tp )) continue;
	if (tp->charge()==0) continue;
	st++;
	h_ptSIM[w]->Fill(sqrt(tp->momentum().perp2()));
	h_etaSIM[w]->Fill(tp->momentum().eta());
	h_vertposSIM[w]->Fill(sqrt(tp->vertex().perp2()));
	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (fabs(tp->momentum().eta())>etaintervals[w][f]&&
	      fabs(tp->momentum().eta())<etaintervals[w][f+1]) {
	    totSIMeta[w][f]++;
	    std::vector<std::pair<reco::TrackRef, double> > rt;
	    if(simRecColl.find(tp) != simRecColl.end()){
	      rt = simRecColl[tp];
	      if (rt.size()!=0) {
		reco::TrackRef t = rt.begin()->first;
		//if ( !selectRecoTracks( *t ) ) continue;//FIXME? TRY WITH SECOND?
		ats++;
		totASSeta[w][f]++;
		edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st << " with pt=" << t->pt() 
						   << " associated with quality:" << rt.begin()->second <<"\n";
	      }
	    }else{
	      edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st
						 << " with pt=" << sqrt(tp->momentum().perp2())
						 << " NOT associated to any reco::Track" << "\n";
	    }
	  }
	} // END for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	
	for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
          if (sqrt(tp->momentum().perp2())>pTintervals[w][f]&&
              sqrt(tp->momentum().perp2())<pTintervals[w][f+1]) {
            totSIMpT[w][f]++;
	    std::vector<std::pair<reco::TrackRef, double> > rt;
            if(simRecColl.find(tp) != simRecColl.end()){
              rt = simRecColl[tp];
	      if (rt.size()!=0) {
		reco::TrackRef t = rt.begin()->first;
		//if ( !selectRecoTracks( *t ) ) continue;//FIXME? TRY WITH SECOND?
		totASSpT[w][f]++;
	      }
	    }
          }
        } // END for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
      }
      if (st!=0) h_tracksSIM[w]->Fill(st);
      

      //
      //fill reconstructed track histograms
      // 
      edm::LogVerbatim("TrackValidator") << "\n# of reco::Tracks with "
					 << label[www].process()<<":"
					 << label[www].label()<<":"
					 << label[www].instance()
					 << " (before cuts): " << tC.size() << "\n";
      int at=0;
      int rT=0;
      for(reco::TrackCollection::size_type i=0; i<tC.size(); ++i){
	reco::TrackRef track(trackCollection, i);
	//if ( !selectRecoTracks( *track ) ) continue;
	rT++;

	std::vector<std::pair<TrackingParticleRef, double> > tp;
	//Compute fake rate vs eta
	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (fabs(track->momentum().eta())>etaintervals[w][f]&&
	      fabs(track->momentum().eta())<etaintervals[w][f+1]) {
	    totRECeta[w][f]++; 
	    if(recSimColl.find(track) != recSimColl.end()){
	      tp = recSimColl[track];
	      if (tp.size()!=0) {
		//if (!selectTPs4FakeRate( *tp.begin()->first )) continue;//FIXME? TRY WITH SECOND?
		totASS2eta[w][f]++;
		edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt() 
						   << " associated with quality:" << tp.begin()->second <<"\n";
	      }
	    }else{
	      edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt()
						 << " NOT associated to any TrackingParticle" << "\n";
	    }
	  }
	}

        for (unsigned int f=0; f<pTintervals[w].size()-1; f++){
          if (sqrt(track->momentum().perp2())>pTintervals[w][f]&&
              sqrt(track->momentum().perp2())<pTintervals[w][f+1]) {
            totRECpT[w][f]++; 
	    if(recSimColl.find(track) != recSimColl.end()){
              tp = recSimColl[track];
	      if (tp.size()!=0) {
		//if (!selectTPs4FakeRate( *tp.begin()->first )) continue;//FIXME? TRY WITH SECOND?
		at++;
		totASS2pT[w][f]++;
	      }
	    }
          }
        }

	//Fill other histos
 	try{
	  if (tp.size()==0) continue;
	
	  TrackingParticleRef tpr = tp.begin()->first;
	  const SimTrack * assocTrack = &(*tpr->g4Track_begin());
	
	  if (associators[ww]=="TrackAssociatorByChi2"){
	    //association chi2
	    double assocChi2 = -tp.begin()->second;//in association map is stored -chi2
	    h_assochi2[www]->Fill(assocChi2);
	    h_assochi2_prob[www]->Fill(chisquared_prob((assocChi2)*5,5));
	  }
    
	  //nchi2 and hits global distributions
	  h_nchi2[w]->Fill(track->normalizedChi2());
	  h_nchi2_prob[w]->Fill(chisquared_prob(track->chi2(),track->ndof()));
	  h_hits[w]->Fill(track->numberOfValidHits());
	  chi2_vs_nhits[w]->Fill(track->numberOfValidHits(),track->normalizedChi2());
	  h_charge[w]->Fill( track->charge() );
	

	  // eta residue; pt, k, theta, phi0, d0, dz pulls
	  Basic3DVector<double> momAtVtx(assocTrack->momentum().x(),assocTrack->momentum().y(),assocTrack->momentum().z());
	  Basic3DVector<double> vert = (Basic3DVector<double>) tpr->vertex();

	  reco::TrackBase::ParameterVector sParameters=
	    associatorForParamAtPca->parametersAtClosestApproachGeom(vert, momAtVtx, track->charge());

	  double qoverpSim = sParameters[0];
	  double lambdaSim = sParameters[1];
	  double phiSim    = sParameters[2];
	  double d0Sim     = -sParameters[3];
	  double dzSim     = sParameters[4]*momAtVtx.mag()/momAtVtx.perp();

	  double qoverpPull=(track->qoverp()-qoverpSim)/track->qoverpError();
	  double thetaPull=(track->lambda()-lambdaSim)/track->thetaError();
	  double phi0Pull=(track->phi()-phiSim)/track->phiError();
	  double d0Pull=(track->d0()-d0Sim)/track->d0Error();
	  double dzPull=(track->dz()-dzSim)/track->dzError();

	  double contrib_Qoverp = ((track->qoverp()-qoverpSim)/track->qoverpError())*
	    ((track->qoverp()-qoverpSim)/track->qoverpError())/5;
	  double contrib_d0 = ((track->d0()-d0Sim)/track->d0Error())*((track->d0()-d0Sim)/track->d0Error())/5;
	  double contrib_dz = ((track->dz()-dzSim)/track->dzError())*((track->dz()-dzSim)/track->dzError())/5;
	  double contrib_theta = ((track->lambda()-lambdaSim)/track->thetaError())*
	    ((track->lambda()-lambdaSim)/track->thetaError())/5;
	  double contrib_phi = ((track->phi()-phiSim)/track->phiError())*
	    ((track->phi()-phiSim)/track->phiError())/5;
	  LogTrace("TrackValidatorTEST") << "assocChi2=" << tp.begin()->second << "\n"
					 << "" <<  "\n"
					 << "ptREC=" << track->pt() << "\n"
					 << "etaREC=" << track->eta() << "\n"
					   << "qoverpREC=" << track->qoverp() << "\n"
					 << "d0REC=" << track->d0() << "\n"
					 << "dzREC=" << track->dz() << "\n"
					 << "thetaREC=" << track->theta() << "\n"
					 << "phiREC=" << track->phi() << "\n"
					 << "" <<  "\n"
					 << "qoverpError()=" << track->qoverpError() << "\n"
					 << "d0Error()=" << track->d0Error() << "\n"
					 << "dzError()=" << track->dzError() << "\n"
					 << "thetaError()=" << track->thetaError() << "\n"
					 << "phiError()=" << track->phiError() << "\n"
					 << "" <<  "\n"
					 << "ptSIM=" << assocTrack->momentum().perp() << "\n"
					 << "etaSIM=" << assocTrack->momentum().pseudoRapidity() << "\n"
					 << "qoverpSIM=" << qoverpSim << "\n"
					 << "d0SIM=" << d0Sim << "\n"
					 << "dzSIM=" << dzSim << "\n"
					 << "thetaSIM=" << M_PI/2-lambdaSim << "\n"
					 << "phiSIM=" << phiSim << "\n"
					 << "" << "\n"
					 << "contrib_Qoverp=" << contrib_Qoverp << "\n"
					 << "contrib_d0=" << contrib_d0 << "\n"
					 << "contrib_dz=" << contrib_dz << "\n"
					 << "contrib_theta=" << contrib_theta << "\n"
					 << "contrib_phi=" << contrib_phi << "\n"
					 << "" << "\n"
					 <<"chi2PULL="<<contrib_Qoverp+contrib_d0+contrib_dz+contrib_theta+contrib_phi<<"\n";
	  
	  h_pullQoverp[w]->Fill(qoverpPull);
	  h_pullTheta[w]->Fill(thetaPull);
	  h_pullPhi0[w]->Fill(phi0Pull);
	  h_pullD0[w]->Fill(d0Pull);
	  h_pullDz[w]->Fill(dzPull);

	  double ptres=track->pt()-assocTrack->momentum().perp(); 
	  double etares=track->eta()-assocTrack->momentum().pseudoRapidity();
	  double ptError =  track->ptError();
	  h_pt[w]->Fill(ptres/ptError);
	  h_eta[w]->Fill(etares);
	  etares_vs_eta[w]->Fill(fabs(track->eta()),etares);

	  //chi2 and #hit vs eta: fill 2D histos
	  chi2_vs_eta[w]->Fill(fabs(track->eta()),track->normalizedChi2());
	  nhits_vs_eta[w]->Fill(fabs(track->eta()),track->numberOfValidHits());
	  nlosthits_vs_eta[w]->Fill(fabs(track->eta()),track->numberOfLostHits());

	  //resolution of track params: fill 2D histos
	  d0res_vs_eta[w]->Fill(fabs(track->eta()),track->d0()-d0Sim);
	  ptres_vs_eta[w]->Fill(fabs(track->eta()),(track->pt()-assocTrack->momentum().perp())/track->pt());
	  z0res_vs_eta[w]->Fill(fabs(track->eta()),track->dz()-dzSim);
	  phires_vs_eta[w]->Fill(fabs(track->eta()),track->phi()-phiSim);
	  cotThetares_vs_eta[w]->Fill(fabs(track->eta()),1/tan(1.570796326794896558-track->lambda())-1/tan(1.570796326794896558-lambdaSim));         

	  //same as before but vs pT
	  d0res_vs_pt[w]->Fill(track->pt(),track->d0()-d0Sim);
	  ptres_vs_pt[w]->Fill(track->pt(),(track->pt()-assocTrack->momentum().perp())/track->pt());
	  z0res_vs_pt[w]->Fill(track->pt(),track->dz()-dzSim);
	  phires_vs_pt[w]->Fill(track->pt(),track->phi()-phiSim);
	  cotThetares_vs_pt[w]->Fill(track->pt(),1/tan(1.570796326794896558-track->lambda())-1/tan(1.570796326794896558-lambdaSim));
  	 
  	 
	  //pulls of track params vs eta: fill 2D histos
	  d0pull_vs_eta[w]->Fill(fabs(track->eta()),d0Pull);
	  ptpull_vs_eta[w]->Fill(fabs(track->eta()),ptres/ptError);
	  z0pull_vs_eta[w]->Fill(fabs(track->eta()),dzPull);
	  phipull_vs_eta[w]->Fill(fabs(track->eta()),phi0Pull);
	  thetapull_vs_eta[w]->Fill(fabs(track->eta()),thetaPull);

	} catch (cms::Exception e){
	  edm::LogError("TrackValidator") << "exception found: " << e.what() << "\n";
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
      TH2F* d0res_eta = new TH2F("d0res_eta","d0res_eta",nint,min,max, 100, -0.01, 0.01);
      copy2D(d0res_eta,d0res_vs_eta[w]);
      FitSlicesYTool fsyt_d0(d0res_eta);
      fsyt_d0.getFittedSigmaWithError(h_d0rmsh[w]);
      delete d0res_eta;
      TH2F* d0res_pt = new TH2F("d0res_pt","d0res_pt",nintpT,minpT,maxpT, 100, -0.01, 0.01);
      copy2D(d0res_pt,d0res_vs_pt[w]);
      FitSlicesYTool fsyt_d0Pt(d0res_pt);
      fsyt_d0Pt.getFittedSigmaWithError(h_d0rmshPt[w]);
      delete d0res_pt;      
      TH2F* ptres_eta = new TH2F("ptres_eta","ptres_eta",nint,min,max, 100, -0.1, 0.1);
      copy2D(ptres_eta,ptres_vs_eta[w]);
      FitSlicesYTool fsyt_pt(ptres_eta);
      fsyt_pt.getFittedSigmaWithError(h_ptrmsh[w]);
      delete ptres_eta;
      TH2F* ptres_pt = new TH2F("ptres_pt","ptres_pt",nintpT,minpT,maxpT, 100, -0.1, 0.1);
      copy2D(ptres_pt,ptres_vs_pt[w]);
      FitSlicesYTool fsyt_ptPt(ptres_pt);
      fsyt_ptPt.getFittedSigmaWithError(h_ptrmshPt[w]);
      delete ptres_pt;
      TH2F* z0res_eta = new TH2F("z0res_eta","z0res_eta",nint,min,max, 150, -0.05, 0.05);
      copy2D(z0res_eta,z0res_vs_eta[w]);
      FitSlicesYTool fsyt_z0(z0res_eta);
      fsyt_z0.getFittedSigmaWithError(h_z0rmsh[w]);
      delete z0res_eta;
      TH2F* z0res_pt = new TH2F("z0res_pt","z0res_pt",nintpT,minpT,maxpT, 150, -0.05, 0.05);
      copy2D(z0res_pt,z0res_vs_pt[w]);
      FitSlicesYTool fsyt_z0Pt(z0res_pt);
      fsyt_z0Pt.getFittedSigmaWithError(h_z0rmshPt[w]);
      delete z0res_pt;
      TH2F* phires_eta = new TH2F("phires_eta","phires_eta",nint,min,max, 100, -0.003, 0.003);
      copy2D(phires_eta,phires_vs_eta[w]);
      FitSlicesYTool fsyt_phi(phires_eta);
      fsyt_phi.getFittedSigmaWithError(h_phirmsh[w]);
      delete phires_eta;
      TH2F* phires_pt = new TH2F("phires_pt","phires_pt",nintpT,minpT,maxpT, 100, -0.003, 0.003);
      copy2D(phires_pt,phires_vs_pt[w]);
      FitSlicesYTool fsyt_phiPt(phires_pt);
      fsyt_phiPt.getFittedSigmaWithError(h_phirmshPt[w]);
      delete phires_pt;
      TH2F* cotThetares_eta = new TH2F("cotThetares_eta","cotThetares_eta",nint,min,max, 120, -0.01, 0.01);
      copy2D(cotThetares_eta,cotThetares_vs_eta[w]);
      FitSlicesYTool fsyt_cotTheta(cotThetares_eta);
      fsyt_cotTheta.getFittedSigmaWithError(h_cotThetarmsh[w]);
      delete cotThetares_eta;
      TH2F* cotThetares_pt = new TH2F("cotThetares_pt","cotThetares_pt",nintpT,minpT,maxpT, 120, -0.01, 0.01);
      copy2D(cotThetares_pt,cotThetares_vs_pt[w]);
      FitSlicesYTool fsyt_cotThetaPt(cotThetares_pt);
      fsyt_cotThetaPt.getFittedSigmaWithError(h_cotThetarmshPt[w]);
      delete cotThetares_pt;

      //chi2 and #hit vs eta: get mean from 2D histos
      TH2F* chi2_eta = new TH2F("chi2_eta","chi2_eta",nint,min,max, 200, 0, 20 );
      copy2D(chi2_eta,chi2_vs_eta[w]);
      doProfileX(chi2_eta,h_chi2meanh[w]);
      delete chi2_eta;
      TH2F* nhits_eta = new TH2F("nhits_eta","nhits_eta",nint,min,max,25,0,25);
      copy2D(nhits_eta,nhits_vs_eta[w]);
      doProfileX(nhits_eta,h_hits_eta[w]);    
      delete nhits_eta;
   
      //pulls of track params vs eta: get sigma from 2D histos
      TH2F* d0pull_eta = new TH2F("d0pull_vs_eta","d0pull_vs_eta",nint,min,max,100,-10,10);
      copy2D(d0pull_eta,d0pull_vs_eta[w]);
      FitSlicesYTool fsyt_d0p(d0pull_eta);
      fsyt_d0p.getFittedSigmaWithError(h_d0pulleta[w]);
      delete d0pull_eta;
      TH2F* ptpull_eta = new TH2F("ptpull_vs_eta","ptpull_vs_eta",nint,min,max,100,-10,10); 
      copy2D(ptpull_eta,ptpull_vs_eta[w]);
      FitSlicesYTool fsyt_ptp(ptpull_eta);
      fsyt_ptp.getFittedSigmaWithError(h_ptpulleta[w]);
      delete ptpull_eta;
      TH2F* z0pull_eta = new TH2F("z0pull_vs_eta","z0pull_vs_eta",nint,min,max,100,-10,10); 
      copy2D(z0pull_eta,z0pull_vs_eta[w]);
      FitSlicesYTool fsyt_z0p(z0pull_eta);
      fsyt_z0p.getFittedSigmaWithError(h_z0pulleta[w]);
      delete z0pull_eta;
      TH2F* phipull_eta = new TH2F("phipull_vs_eta","phipull_vs_eta",nint,min,max,100,-10,10); 
      copy2D(phipull_eta,phipull_vs_eta[w]);
      FitSlicesYTool fsyt_phip(phipull_eta);
      fsyt_phip.getFittedSigmaWithError(h_phipulleta[w]);
      delete phipull_eta;
      TH2F* thetapull_eta = new TH2F("thetapull_vs_eta","thetapull_vs_eta",nint,min,max,100,-10,10);
      copy2D(thetapull_eta,thetapull_vs_eta[w]);
      FitSlicesYTool fsyt_thetap(thetapull_eta);
      fsyt_thetap.getFittedSigmaWithError(h_thetapulleta[w]);
      delete thetapull_eta;
      
      //fill efficiency plot vs eta
      double eff,err;
      for (unsigned int j=0; j<totASSeta[w].size(); j++){
        if (totSIMeta[w][j]!=0){
          eff = ((double) totASSeta[w][j])/((double) totSIMeta[w][j]);
	  err = sqrt(eff*(1-eff)/((double) totSIMeta[w][j]));
          h_effic[w]->setBinContent(j+1, eff);
          h_effic[w]->setBinError(j+1,err);
        }
        else {
          h_effic[w]->setBinContent(j+1, 0);
        }
      }
      //fill efficiency plot vs pt
      for (unsigned int j=0; j<totASSpT[w].size(); j++){
	if (totSIMpT[w][j]!=0){
	  eff = ((double) totASSpT[w][j])/((double) totSIMpT[w][j]);
	  err = sqrt(eff*(1-eff)/((double) totSIMpT[w][j]));
	  h_efficPt[w]->setBinContent(j+1, eff);
	  h_efficPt[w]->setBinError(j+1,err);
	}
	else {
	  h_efficPt[w]->setBinContent(j+1, 0);
	}
      }
      
      //fill fakerate plot
      double frate,ferr;
      for (unsigned int j=0; j<totASS2eta[w].size(); j++){
        if (totRECeta[w][j]!=0){
          frate = 1-((double) totASS2eta[w][j])/((double) totRECeta[w][j]);
	  ferr = sqrt( frate*(1-frate)/(double) totRECeta[w][j] );
          h_fakerate[w]->setBinContent(j+1, frate);
	  h_fakerate[w]->setBinError(j+1,ferr);
        }
        else {
          h_fakerate[w]->setBinContent(j+1, 0);
        }
      }

      for (unsigned int j=0; j<totRECeta[w].size(); j++){
	h_recoeta[w]->setBinContent(j+1, totRECeta[w][j]);
      }
      for (unsigned int j=0; j<totSIMeta[w].size(); j++){
	h_simuleta[w]->setBinContent(j+1, totSIMeta[w][j]);
      }
      for (unsigned int j=0; j<totASSeta[w].size(); j++){
	h_assoceta[w]->setBinContent(j+1, totASSeta[w][j]);
      }
      for (unsigned int j=0; j<totASS2eta[w].size(); j++){
	h_assoc2eta[w]->setBinContent(j+1, totASS2eta[w][j]);
      }

      for (unsigned int j=0; j<totRECpT[w].size(); j++){
        h_recopT[w]->setBinContent(j+1, totRECpT[w][j]);
      }
      for (unsigned int j=0; j<totSIMpT[w].size(); j++){
        h_simulpT[w]->setBinContent(j+1, totSIMpT[w][j]);
      }
      for (unsigned int j=0; j<totASSpT[w].size(); j++){
        h_assocpT[w]->setBinContent(j+1, totASSpT[w][j]);
      }
      for (unsigned int j=0; j<totASS2pT[w].size(); j++){
        h_assoc2pT[w]->setBinContent(j+1, totASS2pT[w][j]);
      }   

      w++;
    }
  }
  if ( out.size() != 0 && dbe_ ) dbe_->save(out);
}




