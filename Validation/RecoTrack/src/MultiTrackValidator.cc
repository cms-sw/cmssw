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
      vector<MonitorElement*>  ptdistribv;
      vector<MonitorElement*>  d0distribv;
  
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
	title <<"p_{t} residue "<< d-step << "<#eta<"<<d;
	ptdistribv.push_back(dbe_->book1D(name.str().c_str(),title.str().c_str(), 200, -2, 2 ));
	name.str("");
	title.str("");
	name <<"d0["<<d-step<<","<<d<<"]";
	title <<"d0 residue "<< d-step << "<d0<"<<d;
	d0distribv.push_back(dbe_->book1D(name.str().c_str(),title.str().c_str(), 200, -0.2, 0.2 ));
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
      h_ptrmsh.push_back( dbe_->book1D("PtRMS","PtRMS vs #eta",nint,min,max) );
      h_d0rmsh.push_back( dbe_->book1D("d0RMS","d0RMS vs #eta",nint,min,max) );
      h_hits_eta.push_back( dbe_->book1D("hits_eta","hits_eta",nint,min,max) );
      
      h_eta.push_back( dbe_->book1D("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
      h_pt.push_back( dbe_->book1D("pullPt", "pull of p_{t}", 100, -10, 10 ) );
      h_pullTheta.push_back( dbe_->book1D("pullTheta","pull of theta parameter",100,-10,10) );
      h_pullPhi0.push_back( dbe_->book1D("pullPhi0","pull of phi0 parameter",100,-10,10) );
      h_pullD0.push_back( dbe_->book1D("pullD0","pull of d0 parameter",100,-10,10) );
      h_pullDz.push_back( dbe_->book1D("pullDz","pull of dz parameter",100,-10,10) );
      h_pullK.push_back( dbe_->book1D("pullK","pull of k parameter",100,-10,10) );
      
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

  LogDebug("TrackValidator") << "\n====================================================" << "\n"
			       << "Analyzing new event" << "\n"
			       << "====================================================\n" << "\n";

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  event.getByLabel("trackingtruth","TrackTruth",TPCollectionH);
  const TrackingParticleCollection tPC = *(TPCollectionH.product());
  
  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      //
      //get collections from the event
      //
      edm::Handle<reco::TrackCollection> trackCollection;
      event.getByLabel(label[www], trackCollection);
      const reco::TrackCollection tC = *(trackCollection.product());
      
      //associate tracks
      reco::RecoToSimCollection recSimColl=associator[ww]->associateRecoToSim(trackCollection,
									      TPCollectionH,
									      &event);
      reco::SimToRecoCollection simRecColl=associator[ww]->associateSimToReco(trackCollection,
									      TPCollectionH, 
									      &event);

      //
      //fill simulation histograms
      //compute number of tracks per eta interval
      //
      LogDebug("TrackValidator") << "\n# of TrackingParticless with " << label[www].c_str()  
				 << ": " << tPC.size() << "\n";
      int st=0;
      for (TrackingParticleCollection::size_type i=0; i<tPC.size(); i++){
	TrackingParticleRef tp(TPCollectionH, i);
	if (fabs(tp->momentum().eta())>max || 
	    fabs(tp->momentum().eta())<min) continue;
	if (sqrt(tp->momentum().perp2())<minpt) continue;
	if ((fabs(tp->parentVertex()->position().perp())/10)>3.5) continue;
	if ((fabs(tp->parentVertex()->position().z())/10)>30) continue;
	int type = tp->g4Track_begin()->product()->begin()->type();
	if (abs(type)!=13&&abs(type)!=11&&abs(type)!=211&&abs(type)!=321&&abs(type)!=2212) continue;
	LogDebug("TrackValidator") << "tp->charge(): " << tp->charge()
				   << "tp->trackPSimHit().size(): " << tp->trackPSimHit().size() 
				   << "\n";
	st++;
	h_ptSIM[w]->Fill(sqrt(tp->momentum().perp2()));
	h_etaSIM[w]->Fill(tp->momentum().eta());
	h_vertposSIM[w]->Fill(sqrt(tp->vertex().perp2()));
	int f=0;
	for (vector<double>::iterator h=etaintervals[w].begin(); h!=etaintervals[w].end()-1; h++){
	  if (fabs(tp->momentum().eta())>etaintervals[w][f]&&
	      fabs(tp->momentum().eta())<etaintervals[w][f+1]) {
	    LogDebug("TrackValidator") << "TrackingParticle with eta: " << tp->momentum().eta() << "\n"
				       << "TrackingParticle with pt : " << sqrt(tp->momentum().perp2()) <<"\n" ;
	    totSIM[w][f]++;
	    std::vector<std::pair<reco::TrackRef, double> > rt;
	    try {
	      rt = simRecColl[tp];
	    } catch (cms::Exception e) {
	      edm::LogError("TrackValidator") << "No reco::Track associated" << "\n";
	    }
	    LogDebug("TrackValidator") << "TrackingParticle number " << st << " associated to " 
				       << rt.size()  << " reco::Track" << "\n";
	    if (rt.size()!=0) {
	      totASS[w][f]++;
	      reco::TrackRef t = rt.begin()->first;
	      hitseta[w][f]+=t->numberOfValidHits();
	    }
	  }
	  f++;
	}
      }
      if (st!=0) h_tracksSIM[w]->Fill(st);
      

      //
      //fill reconstructed track histograms
      // 
      LogDebug("TrackValidator") << "\n# of reco::Tracks with " << label[www].c_str()  
				 << ": " << tC.size() << "\n";
      int at=0;
      int rT=0;
      for(reco::TrackCollection::size_type i=0; i<tC.size(); ++i){
	reco::TrackRef track(trackCollection, i);
	if (fabs(track->eta())>max || fabs(track->eta())<min) continue;
	if (track->pt() < minpt) continue;
	if (track->d0()>3.5) continue;
	if (track->dz()>30) continue;

	rT++;

	std::vector<std::pair<TrackingParticleRef, double> > tp;
	//Compute fake rate vs eta
	for (unsigned int f=0; f<etaintervals[w].size(); f++){
	  if (fabs(track->momentum().eta())>etaintervals[w][f]&&
	      fabs(track->momentum().eta())<etaintervals[w][f+1]) {
	    totREC[w][f]++;
	    LogDebug("TrackValidator") << "Adding Reconstructed Track with eta=" 
				       << track->momentum().eta() << " pt=" << track->pt() << "\n";
	    try {
	      tp = recSimColl[track];
	    } catch (cms::Exception e) {
	      edm::LogError("TrackValidator") << "No TrackingParticle associated" << "\n";
	    }
	    if (tp.size()!=0) {
	      totASS2[w][f]++;
	      LogDebug("TrackValidator") << "Found Associated Track \n";
	    }
	  }
	}

	//Fill other histos
 	try{
// 	  std::vector<std::pair<TrackingParticleRef, double> > tp;
// 	  try {
// 	    tp = recSimColl[track];
// 	  } catch (cms::Exception e) {
// 	    edm::LogError("TrackValidator") << "No TrackingParticle associated" << "\n";
// 	  }

	  LogDebug("TrackValidator") << "reco::Track number " << at << " associated to " 
				     << tp.size()  << " TrackingParticle" << "\n";
	  if (tp.size()==0) continue;
	
	  at++;

	  TrackingParticleRef tpr = tp.begin()->first;
	  SimTrackRefVector::iterator it=tpr->g4Track_begin();
	  const SimTrack * assocTrack = &(**it);
	
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

	  //not needed in 110
	  vert/=10;
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

	  h_pullK[w]->Fill(kres);
	  h_pullTheta[w]->Fill(thetares);
	  h_pullPhi0[w]->Fill(phi0res);
	  h_pullD0[w]->Fill(d0res);
	  h_pullDz[w]->Fill(dzres);

	  double ptres=track->pt()-assocTrack->momentum().perp(); 
	  double etares=track->eta()-assocTrack->momentum().pseudoRapidity();
	
	  h_pt[w]->Fill(ptres/(track->transverseCurvatureError()
			       /track->transverseCurvature()*track-> pt()));
	  h_eta[w]->Fill(etares);
	  ptres_vs_eta[w]->Fill(track->eta(),ptres);
	  etares_vs_eta[w]->Fill(track->eta(),etares);
	
	  //pt residue distribution per eta interval
	  int i=0;
	  for (vector<MonitorElement*>::iterator h=ptdistrib[w].begin(); h!=ptdistrib[w].end(); h++){
	    if (fabs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
		fabs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	      (*h)->Fill(track->pt()-assocTrack->momentum().perp());
	    }
	    i++;
	  }
	
	  //d0 residue distribution per eta interval
	  i=0;
	  for (vector<MonitorElement*>::iterator h=d0distrib[w].begin(); h!=d0distrib[w].end(); h++){
	    if (fabs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
		fabs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	      (*h)->Fill(track->d0()-d0Sim);
	    }
	    i++;
	  }
	} catch (cms::Exception e){
	  edm::LogError("TrackValidator") << "exception found: " << e.what() << "\n";
	}
	LogDebug("TrackValidator") << "end of reco::Track number " << at-1 << "\n";
      }
      if (at!=0) h_tracks[w]->Fill(at);
      h_fakes[w]->Fill(rT-at);
      LogDebug("TrackValidator") << "Total Reconstructed: " << rT << "\n";
      LogDebug("TrackValidator") << "Total Associated: " << at << "\n";
      LogDebug("TrackValidator") << "Total Fakes: " << rT-at << "\n";
      nrec_vs_nsim[w]->Fill(rT,st);
      w++;
    }
  }
}

void MultiTrackValidator::endJob() {

  int w=0;
  for (unsigned int ww=0;ww<associators.size();ww++){
    for (unsigned int www=0;www<label.size();www++){
      //fill pt rms plot versus eta and write pt residue distribution per eta interval histo
      int i=0;
      for (vector<MonitorElement*>::iterator h=ptdistrib[w].begin(); h!=ptdistrib[w].end(); h++){
	h_ptrmsh[w]->Fill(etaintervals[w][i+1]-0.00001 ,(*h)->getRMS());
	i++;
      }
      
      //fill d0 rms plot versus eta and write d0 residue distribution per eta interval histo
      i=0;
      for (vector<MonitorElement*>::iterator h=d0distrib[w].begin(); h!=d0distrib[w].end(); h++){
	h_d0rmsh[w]->Fill(etaintervals[w][i+1]-0.00001 ,(*h)->getRMS());
	i++;
      }
      
      //fill efficiency plot
      double eff;
      for (unsigned int j=0; j<totASS[w].size(); j++){
        if (totSIM[w][j]!=0){
          eff = ((double) totASS[w][j])/((double) totSIM[w][j]);
          h_effic[w]->Fill(etaintervals[w][j+1]-0.00001, eff);
          //h_effic[w]->setBinError(j,sqrt((eff*(1-eff))/((double) totASS[w][j])));
        }
        else {
          h_effic[w]->Fill(etaintervals[w][j+1]-0.00001, 0);
        }
      }

      //fill fakerate plot
      double frate;
      for (unsigned int j=0; j<totASS2[w].size(); j++){
        if (totREC[w][j]!=0){
          frate = 1-((double) totASS2[w][j])/((double) totREC[w][j]);
          h_fakerate[w]->Fill(etaintervals[w][j+1]-0.00001, frate);
        }
        else {
          h_fakerate[w]->Fill(etaintervals[w][j+1]-0.00001, 0);
        }
      }
      
      //fill hits vs eta plot
      for (unsigned int rr=0; rr<hitseta[w].size(); rr++){
	if (totASS[w][rr]!=0)
	  h_hits_eta[w]->Fill(etaintervals[w][rr+1]-0.00001,((double)  hitseta[w][rr])/((double) totASS[w][rr]));
	else h_hits_eta[w]->Fill(etaintervals[w][rr+1]-0.00001, 0);
      }
      w++;
    }
  }
  if ( out.size() != 0 && dbe_ ) dbe_->save(out);
}




