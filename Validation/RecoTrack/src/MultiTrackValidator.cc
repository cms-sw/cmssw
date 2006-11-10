#include "Validation/RecoTrack/interface/MultiTrackValidator.h"

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

using namespace ROOT::Math;

void MultiTrackValidator::beginJob( const EventSetup & setup) {

  dbe_->showDirStructure();

  int j=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){

      dbe_->cd();
      string algo = label[www];
      string assoc= associators[ww];
      string dirName = algo.erase(algo.size()-6,algo.size())+"_"+assoc.erase(0,5);
      dbe_->setCurrentFolder(dirName.c_str());

      string subDirName = dirName + "/pt_d0_residues";
      dbe_->setCurrentFolder(subDirName.c_str());
      vector<double> etaintervalsv;
      vector<double> hitsetav;
      vector<int>    totSIMv,totASSv,totASS2v,totRECv;
      vector<TH1F*>  ptdistribv;
      vector<TH1F*>  d0distribv;
  
      double step=(max-min)/nint;
      ostringstream title,name;
      etaintervalsv.push_back(0);
      for (int k=1;k<nint+1;k++) {
	double d=k*step;
	etaintervalsv.push_back(d);
	totSIMv.push_back(0);
	totASSv.push_back(0);
	totASS2v.push_back(0);
	totRECv.push_back(0);
	hitsetav.push_back(0);
	name.str("");
	title.str("");
	name <<"pt["<<d-step<<","<<d<<"]";
	title <<"#deltap_{t}/p_{t} "<< d-step << "<#eta<"<<d;
	ptdistribv.push_back(new TH1F(name.str().c_str(),title.str().c_str(), 200, -2, 2 ));
	name.str("");
	title.str("");
	name <<"d0["<<d-step<<","<<d<<"]";
	title <<"d0 residue "<< d-step << "<d0<"<<d;
	d0distribv.push_back(new TH1F(name.str().c_str(),title.str().c_str(), 200, -0.2, 0.2 ));
      }
      etaintervals.push_back(etaintervalsv);
      totSIM.push_back(totSIMv);
      totASS.push_back(totASSv);
      totASS2.push_back(totASS2v);
      totREC.push_back(totRECv);
      hitseta.push_back(hitsetav);
      ptdistrib.push_back(ptdistribv);
      d0distrib.push_back(d0distribv);

      dbe_->goUp();
      subDirName = dirName + "/simulation";
      dbe_->setCurrentFolder(subDirName.c_str());
      h_ptSIM.push_back( dbe_->book1D("ptSIM", "generated p_{t}", 5500, 0, 110 ) );
      h_etaSIM.push_back( dbe_->book1D("etaSIM", "generated pseudorapidity", 500, 0, 5 ) );
      h_tracksSIM.push_back( dbe_->book1D("tracksSIM","number of simluated tracks",100,-0.5,99.5) );
      h_vertposSIM.push_back( dbe_->book1D("vertposSIM","Transverse position of sim vertices",1000,-0.5,10000.5) );
      
      dbe_->cd();
      dbe_->setCurrentFolder(dirName.c_str());
      h_tracks.push_back( dbe_->book1D("tracks","number of reconstructed tracks",20,-0.5,19.5) );
      h_fakes.push_back( dbe_->book1D("fakes","number of fake reco tracks",20,-0.5,19.5) );
      h_charge.push_back( dbe_->book1D("charge","charge",3,-1.5,1.5) );
      h_hits.push_back( dbe_->book1D("hits", "number of hits per track", 30, -0.5, 29.5 ) );
      h_nchi2.push_back( dbe_->book1D("chi2", "normalized chi2", 200, 0, 20 ) );
      h_nchi2_prob.push_back( dbe_->book1D("chi2_prob", "normalized chi2 probability",100,0,1));

      h_effic.push_back( dbe_->book1D("effic","efficiency vs #eta",nint,min,max) );
      h_fakerate.push_back( dbe_->book1D("fakerate","fake rate vs #eta",nint,min,max) );
      h_reco.push_back( dbe_->book1D("num_reco","N of reco track vs eta",nint,min,max) );
      h_assoc.push_back( dbe_->book1D("num_assoc(simToReco)","N of associated tracks (simToReco) vs eta",nint,min,max) );
      h_assoc2.push_back( dbe_->book1D("num_assoc(recoToSim)","N of associated (recoToSim) tracks vs eta",nint,min,max) );
      h_simul.push_back( dbe_->book1D("num_simul","N of simulated tracks vs eta",nint,min,max) );
      h_ptrmsh.push_back( dbe_->book1D("sigmaPt/Pt","#singmaPt/Pt vs #eta",nint,min,max) );
      h_d0rmsh.push_back( dbe_->book1D("sigmad0","#sigmad0 vs #eta",nint,min,max) );
      h_hits_eta.push_back( dbe_->book1D("hits_eta","hits_eta",nint,min,max) );
      
      h_eta.push_back( dbe_->book1D("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
      h_pt.push_back( dbe_->book1D("pullPt", "pull of p_{t}", 100, -10, 10 ) );
      h_pullTheta.push_back( dbe_->book1D("pullTheta","pull of theta parameter",250,-25,25) );
      h_pullPhi0.push_back( dbe_->book1D("pullPhi0","pull of phi0 parameter",250,-25,25) );
      h_pullD0.push_back( dbe_->book1D("pullD0","pull of d0 parameter",250,-25,25) );
      h_pullDz.push_back( dbe_->book1D("pullDz","pull of dz parameter",250,-25,25) );
      h_pullK.push_back( dbe_->book1D("pullK","pull of k parameter",250,-25,25) );
      
      if (associators[ww]=="TrackAssociatorByChi2"){
	h_assochi2.push_back( dbe_->book1D("assocChi2","track association chi2",1000000,0,100000) );
	h_assochi2_prob.push_back(dbe_->book1D("assocChi2_prob","probability of association chi2",100,0,1));
      }

      chi2_vs_nhits.push_back( dbe_->book2D("chi2_vs_nhits","chi2 vs nhits",25,0,25,100,0,10) );
      chi2_vs_eta.push_back( dbe_->book2D("chi2_vs_eta","chi2 vs eta",nint,min,max,100,0,10) );
      nhits_vs_eta.push_back( dbe_->book2D("nhits_vs_eta","nhits vs eta",nint,min,max,25,0,25) );
      ptres_vs_eta.push_back( dbe_->book2D("ptres_vs_eta","ptresidue vs eta",nint,min,max,200,-2,2) );
      etares_vs_eta.push_back( dbe_->book2D("etares_vs_eta","etaresidue vs eta",nint,min,max,200,-0.1,0.1) );
      nrec_vs_nsim.push_back( dbe_->book2D("nrec_vs_nsim","nrec_vs_nsim",20,-0.5,19.5,20,-0.5,19.5) );
      j++;
    }
  }
  edm::ESHandle<TrackAssociatorBase> theAssociator;
  for (unsigned int w=0;w<associators.size();w++) {
    setup.get<TrackAssociatorRecord>().get(associators[w],theAssociator);
    associator.push_back( (TrackAssociatorBase *) theAssociator.product() );
  }
  
  edm::ESHandle<TrackAssociatorBase> theAssociatorForParamAtPca;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2",theAssociatorForParamAtPca);
  associatorForParamAtPca = (TrackAssociatorByChi2 *) theAssociatorForParamAtPca.product();
}

void MultiTrackValidator::analyze(const edm::Event& event, const edm::EventSetup& setup){

  edm::LogInfo("TrackValidator") << "\n====================================================" << "\n"
			       << "Analyzing new event" << "\n"
			       << "====================================================\n" << "\n";

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  event.getByLabel("trackingtruth","TrackTruth",TPCollectionH);
  const TrackingParticleCollection tPC = *(TPCollectionH.product());
  
  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      edm::LogVerbatim("TrackValidator") << "Analyzing " << label[www].c_str() << " with " 
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
									      TPCollectionH,
									      &event);
      LogTrace("TrackValidator") << "Calling associateSimToReco method" << "\n";
      reco::SimToRecoCollection simRecColl=associator[ww]->associateSimToReco(trackCollection,
									      TPCollectionH, 
									      &event);

      //
      //fill simulation histograms
      //compute number of tracks per eta interval
      //
      edm::LogVerbatim("TrackValidator") << "\n# of TrackingParticles (before cuts): " << tPC.size() << "\n";
      int ats = 0;
      int st=0;
      for (TrackingParticleCollection::size_type i=0; i<tPC.size(); i++){
	TrackingParticleRef tp(TPCollectionH, i);
	if (!selectTPs4Efficiency( *tp )) continue;
	int type = tp->g4Track_begin()->type();
	if (tp->charge()==0) continue;
	//if (abs(type)!=13&&abs(type)!=11&&abs(type)!=211&&abs(type)!=321&&abs(type)!=2212) continue;
	// 	LogDebug("TrackValidator") << "tp->charge(): " << tp->charge()
	// 				   << "\ntp->trackPSimHit().size(): " << tp->trackPSimHit().size() 
	// 				   << "\n";
	st++;
	h_ptSIM[w]->Fill(sqrt(tp->momentum().perp2()));
	h_etaSIM[w]->Fill(tp->momentum().eta());
	h_vertposSIM[w]->Fill(sqrt(tp->vertex().perp2()));
	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (fabs(tp->momentum().eta())>etaintervals[w][f]&&
	      fabs(tp->momentum().eta())<etaintervals[w][f+1]) {
	    //LogDebug("TrackValidator") << "TrackingParticle with eta: " << tp->momentum().eta() << "\n"
	    //			       << "TrackingParticle with pt : " << sqrt(tp->momentum().perp2()) <<"\n" ;
	    totSIM[w][f]++;
	    std::vector<std::pair<reco::TrackRef, double> > rt;
	    try {
	      rt = simRecColl[tp];
	    } catch (cms::Exception e) {
	      edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st 
						 << " with pt=" << sqrt(tp->momentum().perp2()) 
						 << " NOT associated to any reco::Track" << "\n";
	      edm::LogError("TrackValidator") << e.what() << "\n";
	    }
	    if (rt.size()!=0) {
	      reco::TrackRef t = rt.begin()->first;
 	      if ( !selectTracks4Efficiency( *t ) ) continue;//FIXME TRY WITH SECOND
	      ats++;
	      totASS[w][f]++;
	      hitseta[w][f]+=t->numberOfValidHits();
	      edm::LogVerbatim("TrackValidator") << "TrackingParticle #" << st << " with pt=" << t->pt() 
					 << " associated with quality:" << rt.begin()->second <<"\n";
	    }
	  }
	}
      }
      if (st!=0) h_tracksSIM[w]->Fill(st);
      

      //
      //fill reconstructed track histograms
      // 
      edm::LogVerbatim("TrackValidator") << "\n# of reco::Tracks with " << label[www].c_str()  
					 << "(before cuts): " << tC.size() << "\n";
      int at=0;
      int rT=0;
      for(reco::TrackCollection::size_type i=0; i<tC.size(); ++i){
	reco::TrackRef track(trackCollection, i);
	if ( !selectTracks4FakeRate( *track ) ) continue;
	rT++;

	std::vector<std::pair<TrackingParticleRef, double> > tp;
	//Compute fake rate vs eta
	for (unsigned int f=0; f<etaintervals[w].size()-1; f++){
	  if (fabs(track->momentum().eta())>etaintervals[w][f]&&
	      fabs(track->momentum().eta())<etaintervals[w][f+1]) {
	    totREC[w][f]++;
	    try {
	      tp = recSimColl[track];
	    } catch (cms::Exception e) {
	      edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt() 
					 << " NOT associated to any TrackingParticle" << "\n";
	      edm::LogError("TrackValidator") << e.what() << "\n";
	    }
	    if (tp.size()!=0) {
	      if (!selectTPs4FakeRate( *tp.begin()->first )) continue;//FIXME TRY WITH SECOND
	      totASS2[w][f]++;
	      edm::LogVerbatim("TrackValidator") << "reco::Track #" << rT << " with pt=" << track->pt() 
					 << " associated with quality:" << tp.begin()->second <<"\n";
	    }
	  }
	}

	//Fill other histos
 	try{
	  if (tp.size()==0) continue;
	
	  at++;

	  TrackingParticleRef tpr = tp.begin()->first;
	  //SimTrackRefVector::iterator it=tpr->g4Track_begin();
	  const SimTrack * assocTrack = &(*tpr->g4Track_begin());
	
	  if (associators[ww]=="TrackAssociatorByChi2"){
	    //association chi2
	    double assocChi2 = tp.begin()->second;
	    h_assochi2[www]->Fill(assocChi2);
	    h_assochi2_prob[www]->Fill(chisquared_prob((assocChi2)*5,5));
	  }
    
	  //nchi2 and hits global distributions
	  h_nchi2[w]->Fill(track->normalizedChi2());
	  h_nchi2_prob[w]->Fill(chisquared_prob(track->chi2(),track->ndof()));
	  h_hits[w]->Fill(track->numberOfValidHits());
	  chi2_vs_nhits[w]->Fill(track->numberOfValidHits(),track->normalizedChi2());
	  chi2_vs_eta[w]->Fill(track->eta(),track->normalizedChi2());
	  nhits_vs_eta[w]->Fill(track->eta(),track->numberOfValidHits());
	  h_charge[w]->Fill( track->charge() );
	

	  // eta residue; pt, k, theta, phi0, d0, dz pulls
	  Basic3DVector<double> momAtVtx(assocTrack->momentum().x(),assocTrack->momentum().y(),assocTrack->momentum().z());
	  Basic3DVector<double> vert = (Basic3DVector<double>) tpr->parentVertex()->position();

	  reco::TrackBase::ParameterVector sParameters=
	    associatorForParamAtPca->parametersAtClosestApproachGeom(vert, momAtVtx, track->charge());

	  double kSim     = sParameters[0];
	  double thetaSim = sParameters[1];
	  double phi0Sim  = sParameters[2];
	  double d0Sim    = sParameters[3];
	  double dzSim    = sParameters[4];

	  double kres=(track->transverseCurvature()-kSim)/track->transverseCurvatureError();
	  double thetares=(track->theta()-thetaSim)/track->thetaError();
	  double phi0res=(track->phi0()-phi0Sim)/track->phi0Error();
	  double d0res=(track->d0()-d0Sim)/track->d0Error();
	  double dzres=(track->dz()-dzSim)/track->dzError();
	  if (tp.begin()->second>100){
	    double contrib_K = ((track->transverseCurvature()-kSim)/track->transverseCurvatureError())*
	      ((track->transverseCurvature()-kSim)/track->transverseCurvatureError())/5;
	    double contrib_d0 = ((track->d0()-d0Sim)/track->d0Error())*((track->d0()-d0Sim)/track->d0Error())/5;
	    double contrib_dz = ((track->dz()-dzSim)/track->dzError())*((track->dz()-dzSim)/track->dzError())/5;
	    double contrib_theta = ((track->theta()-thetaSim)/track->thetaError())*
	      ((track->theta()-thetaSim)/track->thetaError())/5;
	    double contrib_phi0 = ((track->phi0()-phi0Sim)/track->phi0Error())*
	      ((track->phi0()-phi0Sim)/track->phi0Error())/5;
	    LogTrace("TrackValidatorTEST") << "assocChi2=" << tp.begin()->second << "\n"
					   << "" <<  "\n"
					   << "ptREC=" << track->pt() << "\n"
					   << "etaREC=" << track->eta() << "\n"
					   << "KREC=" << track->transverseCurvature() << "\n"
					   << "d0REC=" << track->d0() << "\n"
					   << "dzREC=" << track->dz() << "\n"
					   << "thetaREC=" << track->theta() << "\n"
					   << "phi0REC=" << track->phi0() << "\n"
					   << "" <<  "\n"
					   << "transverseCurvatureError()=" << track->transverseCurvatureError() << "\n"
					   << "d0Error()=" << track->d0Error() << "\n"
					   << "dzError()=" << track->dzError() << "\n"
					   << "thetaError()=" << track->thetaError() << "\n"
					   << "phi0Error()=" << track->phi0Error() << "\n"
					   << "" <<  "\n"
					   << "ptSIM=" << assocTrack->momentum().perp() << "\n"
					   << "etaSIM=" << assocTrack->momentum().pseudoRapidity() << "\n"
					   << "kSIM=" << kSim << "\n"
					   << "d0SIM=" << d0Sim << "\n"
					   << "dzSIM=" << dzSim << "\n"
					   << "thetaSIM=" << thetaSim << "\n"
					   << "phi0SIM=" << phi0Sim << "\n"
					   << "" << "\n"
					   << "contrib_K=" << contrib_K << "\n"
					   << "contrib_d0=" << contrib_d0 << "\n"
					   << "contrib_dz=" << contrib_dz << "\n"
					   << "contrib_theta=" << contrib_theta << "\n"
					   << "contrib_phi0=" << contrib_phi0 << "\n"
					   << "" << "\n"
					   <<"chi2PULL="<<contrib_K+contrib_d0+contrib_dz+contrib_theta+contrib_phi0<<"\n";
	  }
	  
	  h_pullK[w]->Fill(kres);
	  h_pullTheta[w]->Fill(thetares);
	  h_pullPhi0[w]->Fill(phi0res);
	  h_pullD0[w]->Fill(d0res);
	  h_pullDz[w]->Fill(dzres);

	  double ptres=track->pt()-assocTrack->momentum().perp(); 
	  double etares=track->eta()-assocTrack->momentum().pseudoRapidity();
	
	  h_pt[w]->Fill(ptres/track->ptError());
	  h_eta[w]->Fill(etares);
	  ptres_vs_eta[w]->Fill(track->eta(),ptres);
	  etares_vs_eta[w]->Fill(track->eta(),etares);
	
	  //pt residue distribution per eta interval
	  int i=0;
	  for (vector<TH1F*>::iterator h=ptdistrib[w].begin(); h!=ptdistrib[w].end(); h++){
	    if (fabs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
		fabs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	      (*h)->Fill( (track->pt()-assocTrack->momentum().perp())/track->pt() );
	    }
	    i++;
	  }
	
	  //d0 residue distribution per eta interval
	  i=0;
	  for (vector<TH1F*>::iterator h=d0distrib[w].begin(); h!=d0distrib[w].end(); h++){
	    if (fabs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
		fabs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	      (*h)->Fill(track->d0()-d0Sim);
	    }
	    i++;
	  }
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

  TF1 * fit;
  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      //fill pt rms plot versus eta and write pt residue distribution per eta interval histo
      int i=0;
      for (vector<TH1F*>::iterator h=ptdistrib[w].begin(); h!=ptdistrib[w].end(); h++){
	fit = new TF1("g","gaus");
	(*h)->Fit("g");
	h_ptrmsh[w]->setBinContent(i+1, fit->GetParameter(2));
	delete fit;
	i++;
      }
      
      //fill d0 rms plot versus eta and write d0 residue distribution per eta interval histo
      i=0;
      for (vector<TH1F*>::iterator h=d0distrib[w].begin(); h!=d0distrib[w].end(); h++){
	fit = new TF1("g","gaus");
	(*h)->Fit("g");
	h_d0rmsh[w]->setBinContent(i+1, fit->GetParameter(2));
	delete fit;
	i++;
      }
      
      //fill efficiency plot
      double eff,err;
      for (unsigned int j=0; j<totASS[w].size(); j++){
        if (totSIM[w][j]!=0){
          eff = ((double) totASS[w][j])/((double) totSIM[w][j]);
	  err = sqrt(eff*(1-eff)/((double) totSIM[w][j]));
	  //	  edm::LogVerbatim("TrackValidatorInfo") 
	  cout
	    << "efficiency in eta interval [" << etaintervals[w][j] << ","
	    << etaintervals[w][j+1] << "] is "
	    << eff << " (" << totASS[w][j] << "/" << totSIM[w][j] << ") +- "
	    << err <<"\n";

          h_effic[w]->setBinContent(j+1, eff);
          h_effic[w]->setBinError(j+1,err);
        }
        else {
          h_effic[w]->setBinContent(j+1, 0);
        }
      }

      //fill fakerate plot
      double frate;
      for (unsigned int j=0; j<totASS2[w].size(); j++){
        if (totREC[w][j]!=0){
          frate = 1-((double) totASS2[w][j])/((double) totREC[w][j]);
          h_fakerate[w]->setBinContent(j+1, frate);
        }
        else {
          h_fakerate[w]->setBinContent(j+1, 0);
        }
      }

      for (unsigned int j=0; j<totREC[w].size(); j++){
	h_reco[w]->setBinContent(j+1, totREC[w][j]);
      }
      for (unsigned int j=0; j<totSIM[w].size(); j++){
	h_simul[w]->setBinContent(j+1, totSIM[w][j]);
      }
      for (unsigned int j=0; j<totASS[w].size(); j++){
	h_assoc[w]->setBinContent(j+1, totASS[w][j]);
      }
      for (unsigned int j=0; j<totASS2[w].size(); j++){
	h_assoc2[w]->setBinContent(j+1, totASS2[w][j]);
      }
      
      //fill hits vs eta plot
      for (unsigned int rr=0; rr<hitseta[w].size(); rr++){
	if (totASS[w][rr]!=0)
	  h_hits_eta[w]->setBinContent(rr+1,((double)  hitseta[w][rr])/((double) totASS[w][rr]));
	else h_hits_eta[w]->setBinContent(rr+1, 0);
      }
      w++;
    }
  }
  if ( out.size() != 0 && dbe_ ) dbe_->save(out);
}




