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
      string dirName = label[www]+associators[ww];
      dbe_->setCurrentFolder(dirName.c_str());
      
      vector<double> etaintervalsv;
      vector<double> hitsetav;
      vector<int>    totSIMv,totRECv;
      vector<MonitorElement*>  ptdistribv;
      vector<MonitorElement*>  etadistribv;
  
      double step=(max-min)/nint;
      ostringstream title,name;
      etaintervalsv.push_back(0);
      for (int k=1;k<nint+1;k++) {
	double d=k*step;
	etaintervalsv.push_back(d);
	totSIMv.push_back(0);
	totRECv.push_back(0);
	hitsetav.push_back(0);
	name.str("");
	title.str("");
	name <<"pt["<<d<<","<<d+step<<"]";
	title <<"p_{t} residue "<< d << "<#eta<"<<d+step;
	ptdistribv.push_back(dbe_->book1D(name.str().c_str(),title.str().c_str(), 200, -2, 2 ));
	name.str("");
	title.str("");
	name <<"eta["<<d<<","<<d+step<<"]";
	title <<"eta residue "<< d << "<#eta<"<<d+step;
	etadistribv.push_back(dbe_->book1D(name.str().c_str(),title.str().c_str(), 200, -0.2, 0.2 ));
      }
      etaintervals.push_back(etaintervalsv);
      totSIM.push_back(totSIMv);
      totREC.push_back(totRECv);
      hitseta.push_back(hitsetav);
      ptdistrib.push_back(ptdistribv);
      etadistrib.push_back(etadistribv);
     
      h_ptSIM.push_back( dbe_->book1D("ptSIM", "generated p_{t}", 5500, 0, 110 ) );
      h_etaSIM.push_back( dbe_->book1D("etaSIM", "generated pseudorapidity", 500, 0, 5 ) );
      h_tracksSIM.push_back( dbe_->book1D("tracksSIM","number of simluated tracks",100,-0.5,99.5) );
      h_vertposSIM.push_back( dbe_->book1D("vertposSIM","Transverse position of sim vertices",1000,-0.5,10000.5) );
      
      //     h_pt     = dbe_->book1D("pt", "p_{t} residue", 2000, -500, 500 );
      h_pt.push_back( dbe_->book1D("pullPt", "pull of p_{t}", 100, -10, 10 ) );
      h_pt2.push_back( dbe_->book1D("pt2", "p_{t} residue (#tracks>1)", 300, -15, 15 ) );
      h_eta.push_back( dbe_->book1D("eta", "pseudorapidity residue", 1000, -0.1, 0.1 ) );
      h_tracks.push_back( dbe_->book1D("tracks","number of reconstructed tracks",20,-0.5,19.5) );
      h_fakes.push_back( dbe_->book1D("fakes","number of fake reco tracks",20,-0.5,19.5) );
      h_nchi2.push_back( dbe_->book1D("nchi2", "normalized chi2", 200, 0, 20 ) );
      h_nchi2_prob.push_back( dbe_->book1D("chi2_prob", "normalized chi2 probability",100,0,1));
      h_hits.push_back( dbe_->book1D("hits", "number of hits per track", 30, -0.5, 29.5 ) );
      h_effic.push_back( dbe_->book1D("effic","efficiency vs #eta",nint,min,max) );
      h_ptrmsh.push_back( dbe_->book1D("PtRMS","PtRMS vs #eta",nint,min,max) );
      h_deltaeta.push_back( dbe_->book1D("etaRMS","etaRMS vs #eta",nint,min,max) );
      h_hits_eta.push_back( dbe_->book1D("hits_eta","hits_eta",nint,min,max) );
//       h_effic.push_back( dbe_->book1D("effic","efficiency vs #eta",nint,&etaintervals[j][0]) );
//       h_ptrmsh.push_back( dbe_->book1D("PtRMS","PtRMS vs #eta",nint,&etaintervals[j][0]) );
//       h_deltaeta.push_back( dbe_->book1D("etaRMS","etaRMS vs #eta",nint,&etaintervals[j][0]) );
//       h_hits_eta.push_back( dbe_->book1D("hits_eta","hits_eta",nint,&etaintervals[j][0]) );
      h_charge.push_back( dbe_->book1D("charge","charge",3,-1.5,1.5) );
      
      h_pullTheta.push_back( dbe_->book1D("pullTheta","pull of theta parameter",100,-10,10) );
      h_pullPhi0.push_back( dbe_->book1D("pullPhi0","pull of phi0 parameter",100,-10,10) );
      h_pullD0.push_back( dbe_->book1D("pullD0","pull of d0 parameter",100,-10,10) );
      h_pullDz.push_back( dbe_->book1D("pullDz","pull of dz parameter",100,-10,10) );
      h_pullK.push_back( dbe_->book1D("pullK","pull of k parameter",100,-10,10) );
      
      chi2_vs_nhits.push_back( dbe_->book2D("chi2_vs_nhits","chi2 vs nhits",25,0,25,100,0,10) );
      chi2_vs_eta.push_back( dbe_->book2D("chi2_vs_eta","chi2 vs eta",nint,min,max,100,0,10) );
      nhits_vs_eta.push_back( dbe_->book2D("nhits_vs_eta","nhits vs eta",nint,min,max,25,0,25) );
      ptres_vs_eta.push_back( dbe_->book2D("ptres_vs_eta","ptresidue vs eta",nint,min,max,200,-2,2) );
      etares_vs_eta.push_back( dbe_->book2D("etares_vs_eta","etaresidue vs eta",nint,min,max,200,-0.1,0.1) );
      nrec_vs_nsim.push_back( dbe_->book2D("nrec_vs_nsim","nrec_vs_nsim",20,-0.5,19.5,20,-0.5,19.5) );
      
      h_assochi2.push_back( dbe_->book1D("assochi2","track association chi2",200,0,50) );
      h_assochi2_prob.push_back(dbe_->book1D("assochi2prob","probability of association chi2",100,0,1));
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
      reco::RecoToSimCollection p = associator[ww]->associateRecoToSim(trackCollection,TPCollectionH, &event);
      reco::SimToRecoCollection q = associator[ww]->associateSimToReco(trackCollection,TPCollectionH, &event);

      //
      //fill simulation histograms
      //compute number of tracks per eta interval
      //
      LogDebug("TrackValidator") << "\n# of TrackingParticless with " << label[www].c_str()  
				 << ": " << tPC.size() << "\n";
      int st=0;
      for (TrackingParticleCollection::size_type i=0; i<tPC.size(); i++){
	TrackingParticleRef tp(TPCollectionH, i);
	if (abs(tp->momentum().eta())>max || 
	    abs(tp->momentum().eta())<min) continue;
	if (sqrt(tp->momentum().perp2())<minpt) continue;
	int type = tp->g4Track_begin()->product()->begin()->type();
	if (abs(type)!=13||abs(type)!=11||abs(type)!=211||abs(type)!=321||abs(type)!=2212) continue;
	LogDebug("TrackValidator") << "PIPPO tp->charge(): " << tp->charge()
				   << "PIPPO tp->trackPSimHit().size(): " << tp->trackPSimHit().size() 
				   << "\n";
	st++;
	h_ptSIM[w]->Fill(sqrt(tp->momentum().perp2()));
	h_etaSIM[w]->Fill(tp->momentum().eta());
	h_vertposSIM[w]->Fill(sqrt(tp->vertex().perp2()));
	int f=0;
	for (vector<double>::iterator h=etaintervals[w].begin(); h!=etaintervals[w].end()-1; h++){
	  if (abs(tp->momentum().eta())>etaintervals[w][f]&&
	      abs(tp->momentum().eta())<etaintervals[w][f+1]) {
	    LogDebug("TrackValidator") << "TrackingParticle with eta: " << tp->momentum().eta() << "\n"
				       << "TrackingParticle with pt : " << sqrt(tp->momentum().perp2()) <<"\n" ;
	    totSIM[w][f]++;
	    std::vector<std::pair<reco::TrackRef, double> > rt;
	    try {
	      rt = q[tp];
	    } catch (cms::Exception e) {
	      edm::LogError("TrackValidator") << "No reco::Track associated" << "\n";
	    }
	    LogDebug("TrackValidator") << "TrackingParticle number " << st << " associated to " 
				       << rt.size()  << " reco::Track" << "\n";
	    if (rt.size()!=0) {
	      totREC[w][f]++;
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
      LogDebug("TrackValidator") << "\n# of reco::Tracks with " << label[www].c_str()  << ": " << tC.size() << "\n";
      int at=0;
      int rT=0;
      for(reco::TrackCollection::size_type i=0; i<tC.size(); ++i){
	reco::TrackRef track(trackCollection, i);
	if (abs(track->eta())>max || abs(track->eta())<min) continue;
	if (track->pt() < minpt) continue;

	rT++;

	try{
	  std::vector<std::pair<TrackingParticleRef, double> > tp;
	  try {
	    tp = p[track];
	  } catch (cms::Exception e) {
	    edm::LogError("TrackValidator") << "No TrackingParticle associated" << "\n";
	  }

	  LogDebug("TrackValidator") << "reco::Track number " << at << " associated to " 
				     << tp.size()  << " TrackingParticle" << "\n";
	  if (tp.size()==0) continue;
	
	  TrackingParticleRef tpr = tp.begin()->first;
	  SimTrackRefVector::iterator it=tpr->g4Track_begin();
	  const SimTrack * assocTrack = &(**it);
	
	  //association chi2
	  double assocChi2 = tp.begin()->second;
	  h_assochi2[w]->Fill(assocChi2);
	  h_assochi2_prob[w]->Fill(chisquared_prob((assocChi2)*5,5));
	
	  at++;
      
	  //nchi2 and hits global distributions
	  h_nchi2[w]->Fill(track->normalizedChi2());
	  h_nchi2_prob[w]->Fill(chisquared_prob(track->chi2(),track->ndof()));
	  h_hits[w]->Fill(track->numberOfValidHits());
	  chi2_vs_nhits[w]->Fill(track->numberOfValidHits(),track->normalizedChi2());
	  chi2_vs_eta[w]->Fill(track->eta(),track->normalizedChi2());
	  nhits_vs_eta[w]->Fill(track->eta(),track->numberOfValidHits());
	  //h_hits_eta[w]->Fill(track->eta(),track->numberOfValidHits());
	  h_charge[w]->Fill( track->charge() );
	

	  // eta residue; pt, k, theta, phi0, d0, dz pulls
	  Basic3DVector<double> momAtVtx(assocTrack->momentum().x(),assocTrack->momentum().y(),assocTrack->momentum().z());
	  Basic3DVector<double> vert = (Basic3DVector<double>) tpr->parentVertex()->position();;

	  //not needed in 110
	  //	  vert/=10;
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

	  // 	LogDebug("TrackValidator") << "dzSim           : " << dzSim << "\n"
	  // 				   << "track->dz()     : " << track->dz() << "\n"
	  // 				   << "track->dzError(): " << track->dzError() << "\n"
	  // 				   << "dzres           : " << dzres << "\n";

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
	    if (abs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
		abs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	      (*h)->Fill(track->pt()-assocTrack->momentum().perp());
	    }
	    i++;
	  }
	
	  //eta residue distribution per eta interval
	  i=0;
	  for (vector<MonitorElement*>::iterator h=etadistrib[w].begin(); h!=etadistrib[w].end(); h++){
	    if (abs(assocTrack->momentum().pseudoRapidity())>etaintervals[w][i]&&
		abs(assocTrack->momentum().pseudoRapidity())<etaintervals[w][i+1]) {
	      (*h)->Fill(etares);
	    }
	    i++;
	  }
	} catch (cms::Exception e){
	  edm::LogError("TrackValidator") << "exception found: " << e.what() << "\n";
	}
	LogDebug("TrackValidator") << "end of reco::Track number " << at-1 << "\n";
      }
      if (at!=0) h_tracks[w]->Fill(at);
      h_fakes[w]->Fill(tC.size()-at);
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
      
      //fill eta rms plot versus eta and write eta residue distribution per eta interval histo
      i=0;
      for (vector<MonitorElement*>::iterator h=etadistrib[w].begin(); h!=etadistrib[w].end(); h++){
	h_deltaeta[w]->Fill(etaintervals[w][i+1]-0.00001 ,(*h)->getRMS());
	i++;
      }
      
      //fill efficiency plot
      for (unsigned int j=0; j<totREC[w].size(); j++){
	if (totSIM[w][j]!=0){
	  h_effic[w]->Fill(etaintervals[w][j+1]-0.00001, ((double) totREC[w][j])/((double) totSIM[w][j]));
	}
	else {
	  h_effic[w]->Fill(etaintervals[w][j+1]-0.00001, 0);
	}
      }
      
      //fill hits vs eta plot
      for (unsigned int rr=0; rr<hitseta[w].size(); rr++){
	if (totREC[w][rr]!=0)
	  h_hits_eta[w]->Fill(etaintervals[w][rr+1]-0.00001,((double)  hitseta[w][rr])/((double) totREC[w][rr]));
	else h_hits_eta[w]->Fill(etaintervals[w][rr+1]-0.00001, 0);
      }
      w++;
    }
  }
  if ( out.size() != 0 && dbe_ ) dbe_->save(out);
}




