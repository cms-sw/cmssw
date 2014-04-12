#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimTracker/TrackAssociation/test/testTrackAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

#include <memory>
#include <iostream>
#include <string>

//class TrackAssociator; 
class TrackAssociatorByHits; 
class TrackerHitAssociator; 

using namespace reco;
using namespace std;
using namespace edm;

testTrackAssociator::testTrackAssociator(edm::ParameterSet const& conf) {
  tracksTag = conf.getParameter< edm::InputTag >("tracksTag");
  tpTag = conf.getParameter< edm::InputTag >("tpTag");
  simtracksTag = conf.getParameter< edm::InputTag >("simtracksTag");
  simvtxTag = conf.getParameter< edm::InputTag >("simvtxTag");
}

testTrackAssociator::~testTrackAssociator() {
}

void testTrackAssociator::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  
  edm::ESHandle<MagneticField> theMF;
  setup.get<IdealMagneticFieldRecord>().get(theMF);
  edm::ESHandle<TrackAssociatorBase> theChiAssociator;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2",theChiAssociator);
  associatorByChi2 = (TrackAssociatorBase *) theChiAssociator.product();
  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  setup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
  associatorByHits = (TrackAssociatorBase *) theHitsAssociator.product();

  Handle<View<Track> > trackCollectionH;
  event.getByLabel(tracksTag,trackCollectionH);
  const View<Track>  tC = *(trackCollectionH.product()); 
  
  Handle<SimTrackContainer> simTrackCollection;
  event.getByLabel(simtracksTag, simTrackCollection);
  const SimTrackContainer simTC = *(simTrackCollection.product());
  
  Handle<SimVertexContainer> simVertexCollection;
  event.getByLabel(simvtxTag, simVertexCollection);
  const SimVertexContainer simVC = *(simVertexCollection.product());

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  event.getByLabel(tpTag,TPCollectionH);
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  cout << "\nEvent ID = "<< event.id() << endl ;


  //RECOTOSIM 
  cout << "                      ****************** Reco To Sim ****************** " << endl;
  cout << "-- Associator by hits --" << endl;  
  reco::RecoToSimCollection p = 
    associatorByHits->associateRecoToSim (trackCollectionH,TPCollectionH,&event,&setup );
  for(View<Track>::size_type i=0; i<tC.size(); ++i) {
    RefToBase<Track> track(trackCollectionH, i);
    try{ 
      std::vector<std::pair<TrackingParticleRef, double> > tp = p[track];
	cout << "Reco Track pT: "  << setw(6) << track->pt() 
	     <<  " matched to " << tp.size() << " MC Tracks" << std::endl;
	for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = tp.begin(); 
	     it != tp.end(); ++it) {
	  TrackingParticleRef tpr = it->first;
	  double assocChi2 = it->second;
	  cout << "\t\tMCTrack " << setw(2) << tpr.index() << " pT: " << setw(6) << tpr->pt() << 
	    " NShared: " << assocChi2 << endl;
	}
    } catch (Exception event) {
      cout << "->   Track pT: " 
	   << setprecision(2) << setw(6) << track->pt() 
	   <<  " matched to 0  MC Tracks" << endl;
    }
  }
  cout << "-- Associator by chi2 --" << endl;  
  p = associatorByChi2->associateRecoToSim (trackCollectionH,TPCollectionH,&event,&setup );
  for(View<Track>::size_type i=0; i<tC.size(); ++i) {
    RefToBase<Track> track(trackCollectionH, i);
    try{ 
      std::vector<std::pair<TrackingParticleRef, double> > tp = p[track];
      cout << "Reco Track pT: "  << setw(6) << track->pt() 
	   <<  " matched to " << tp.size() << " MC Tracks" << std::endl;
      for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = tp.begin(); 
	   it != tp.end(); ++it) {
	TrackingParticleRef tpr = it->first;
	double assocChi2 = it->second;
	cout << "\t\tMCTrack " << setw(2) << tpr.index() << " pT: " << setw(6) << tpr->pt() << 
	  " chi2: " << assocChi2 << endl;
      }
    } catch (Exception event) {
      cout << "->   Track pT: " 
	   << setprecision(2) << setw(6) << track->pt() 
	   <<  " matched to 0  MC Tracks" << endl;
    }
  }
  //SIMTORECO
  cout << "                      ****************** Sim To Reco ****************** " << endl;
  cout << "-- Associator by hits --" << endl;  
  reco::SimToRecoCollection q = 
    associatorByHits->associateSimToReco(trackCollectionH,TPCollectionH,&event,&setup );
  for(SimTrackContainer::size_type i=0; i<simTC.size(); ++i){
    TrackingParticleRef tp (TPCollectionH,i);
    try{ 
      std::vector<std::pair<RefToBase<Track>, double> > trackV = q[tp];
	cout << "Sim Track " << setw(2) << tp.index() << " pT: "  << setw(6) << tp->pt() 
	     <<  " matched to " << trackV.size() << " reco::Tracks" << std::endl;
	for (std::vector<std::pair<RefToBase<Track>,double> >::const_iterator it=trackV.begin(); it != trackV.end(); ++it) {
	  RefToBase<Track> tr = it->first;
	  double assocChi2 = it->second;
	  cout << "\t\treco::Track pT: " << setw(6) << tr->pt() << 
	    " NShared: " << assocChi2 << endl;
	}
    } catch (Exception event) {
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: " 
	   <<setprecision(2)<<setw(6)<<tp->pt() 
	   <<  " matched to 0  reco::Tracks" << endl;
    }
  }
  cout << "-- Associator by chi2 --" << endl;  
  q = associatorByChi2->associateSimToReco(trackCollectionH,TPCollectionH,&event,&setup );
  for(SimTrackContainer::size_type i=0; i<simTC.size(); ++i){
    TrackingParticleRef tp (TPCollectionH,i);
    try{ 
      std::vector<std::pair<RefToBase<Track>, double> > trackV = q[tp];
      cout << "Sim Track " << setw(2) << tp.index() << " pT: "  << setw(6) << tp->pt() 
	   <<  " matched to " << trackV.size() << " reco::Tracks" << std::endl;
      for (std::vector<std::pair<RefToBase<Track>,double> >::const_iterator it=trackV.begin(); it != trackV.end(); ++it) {
	RefToBase<Track> tr = it->first;
	double assocChi2 = it->second;
	cout << "\t\treco::Track pT: " << setw(6) << tr->pt() << 
	  " chi2: " << assocChi2 << endl;
      }
    } catch (Exception event) {
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: " 
	   <<setprecision(2)<<setw(6)<<tp->pt() 
	   <<  " matched to 0  reco::Tracks" << endl;
    }
  }
}



#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(testTrackAssociator);



