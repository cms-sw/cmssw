

using namespace std;


#include "SimTracker/TrackAssociation/test/testTrackAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "Math/GenVector/BitReproducible.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <memory>
#include <iostream>
#include <string>

class TrackAssociator; 
class TrackerHitAssociator; 

using namespace reco;
using namespace std;
using namespace edm;


testTrackAssociator::testTrackAssociator(edm::ParameterSet const& conf) : 
  conf_(conf),
  doPixel_( conf.getParameter<bool>("associatePixel") ),
  doStrip_( conf.getParameter<bool>("associateStrip") ) 
{
  std::cout << " Constructor " << std::endl;
}

testTrackAssociator::~testTrackAssociator()
{
  std::cout << " Destructor " << std::endl;
}

void testTrackAssociator::beginJob(const EventSetup & setup) {

  edm::ESHandle<MagneticField> theMF;
  setup.get<IdealMagneticFieldRecord>().get(theMF);
  //  tassociator = (TrackAssociatorBase *) theAssociator.product();
  //associator = new TrackAssociatorByChi2(theMF);

}

void testTrackAssociator::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  using namespace edm;
  using namespace reco;

  tassociator = new TrackAssociator(event, conf_);
  
  Handle<reco::TrackCollection> trackCollectionH;
  event.getByLabel("ctfWithMaterialTracks",trackCollectionH);
  const  reco::TrackCollection  tC = *(trackCollectionH.product()); 
  
  Handle<SimTrackContainer> simTrackCollection;
  event.getByLabel("g4SimHits", simTrackCollection);
  const SimTrackContainer simTC = *(simTrackCollection.product());
  
  Handle<SimVertexContainer> simVertexCollection;
  event.getByLabel("g4SimHits", simVertexCollection);
  const SimVertexContainer simVC = *(simVertexCollection.product());

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  event.getByLabel("trackingtruth","TrackTruth",TPCollectionH);
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());

  cout << "\nEvent ID = "<< event.id() << endl ;


#if 0

  //Test TrackAssociatorByChi2
  //compareTracksParam
  TrackAssociatorByChi2::RecoToSimPairAssociation g =  associator->compareTracksParam(tC,simTC,simVC);
  for (TrackAssociatorByChi2::RecoToSimPairAssociation::iterator vit=g.begin();vit!=g.end();++vit){
    double chi2 = vit->second.begin()->first;
    reco::Track& rt = vit->first;
    SimTrack& st = vit->second.begin()->second;
    cout << "Chi2 associator - chi2 value: " << chi2 << endl;
    cout << "Chi2 associator - pt residue: " << rt.pt()-st.momentum().perp() << endl;
  }

  //RECOTOSIM 
  reco::RecoToSimCollection p = tassociator->associateRecoToSim (trackCollectionH,TPCollectionH );
  for(TrackCollection::size_type i=0; i<tC.size(); ++i){
    TrackRef track(trackCollectionH, i);
    try{ 
      std::vector<std::pair<TrackingParticleRef, double> > tp = p[track];
      cout << "->   Track " << setw(2) << track.index() << " pT: "  << setw(6) << track->pt() 
	   <<  " matched to " << tp.size() << " MC Tracks" << std::endl;
      for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = tp.begin(); 
	   it != tp.end(); ++it) {
	cout << "1" << endl;
	TrackingParticleRef tpr = it->first;
	cout << "2" << endl;
	double assocChi2 = it->second;
	cout << "   MCTrack " << setw(2) << tpr.index() << " pT: " << setw(6) << tpr->pt() << 
	  " chi2: " << assocChi2 << endl;
      }
    } catch (Exception event) {
      cout << "->   Track " << setw(2) << track.index() << " pT: " << setprecision(2) << setw(6) << track->pt() 
	   <<  " matched to 0  MC Tracks" << endl;
    }
  }

  //SIMTORECO
  reco::SimToRecoCollection q = tassociator->associateSimToReco (trackCollectionH,TPCollectionH );
  for(SimTrackContainer::size_type i=0; i<simTC.size(); ++i){
    TrackingParticleRef tp (TPCollectionH,i);
    try{ 
      std::vector<std::pair<TrackRef, double> > trackV = q[tp];
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: "  << setw(6) << tp->pt() 
	   <<  " matched to " << trackV.size() << " reco::Tracks" << std::endl;
      for (std::vector<std::pair<TrackRef,double> >::const_iterator it=trackV.begin(); it != trackV.end(); ++it) {
	TrackRef tr = it->first;
	double assocChi2 = it->second;
	cout << "   reco::Track " << setw(2) << tr.index() << " pT: " << setw(6) << tr->pt() << 
	  " chi2: " << assocChi2 << endl;
      }
    } catch (Exception event) {
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: " <<setprecision(2)<<setw(6)<<tp->pt() 
	   <<  " matched to 0  reco::Tracks" << endl;
    }
  }

#endif


  //#if 0
  //Test AssociateByHitsRecoTrack
  int minHitFraction = 0;

  if(!doPixel_ && !doStrip_)  throw Exception(errors::Configuration,"Strip and pixel association disabled");
  
  const RecoToSimCollection assocmap = tassociator->associateRecoToSim(trackCollectionH,TPCollectionH );
  //now test map 
  cout << "Found " << assocmap.size() << " matched reco tracks" << std::endl;

  for(TrackCollection::size_type i=0; i<tC.size(); ++i){
    TrackRef track(trackCollectionH, i);
    try{ 
      std::vector<std::pair<TrackingParticleRef, double> > tp = assocmap[track];
      cout << "->   Track " << setw(2) << track.index() << " pT: "  << setw(6) << track->momentum() 
	   <<  " matched to " << tp.size() << " MC Tracks" << std::endl;
      for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator it = tp.begin(); 
	   it != tp.end(); ++it) {
	TrackingParticleRef tpr = it->first;
	double assocChi2 = it->second;
	cout << "   MCTrack " << setw(2) << tpr.index() << " mom: " << setw(6) << tpr->momentum() << 
	  " fraction: " << assocChi2 << endl;
      }
    } catch (Exception event) {
      cout << "->   Track " << setw(2) << track.index() << " mom: " << setprecision(2) << setw(6) << track->pt() 
	   <<  " matched to 0  MC Tracks" << endl;
    }
  }


  //SIMTORECO
  reco::SimToRecoCollection q = tassociator->associateSimToReco (trackCollectionH,TPCollectionH);
  for(SimTrackContainer::size_type i=0; i<simTC.size(); ++i){
    TrackingParticleRef tp (TPCollectionH,i);
    try{ 
      std::vector<std::pair<TrackRef, double> > trackV = q[tp];
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: "  << setw(6) << tp->pt() 
	   <<  " matched to " << trackV.size() << " reco::Tracks" << std::endl;
      for (std::vector<std::pair<TrackRef,double> >::const_iterator it=trackV.begin(); it != trackV.end(); ++it) {
	TrackRef tr = it->first;
	double assocChi2 = it->second;
	cout << "   reco::Track " << setw(2) << tr.index() << " pT: " << setw(6) << tr->pt() << 
	  " fraction: " << assocChi2 << endl;
      }
    } catch (Exception event) {
      cout << "->   TrackingParticle " << setw(2) << tp.index() << " pT: " <<setprecision(2)<<setw(6)<<tp->pt() 
	   <<  " matched to 0  reco::Tracks" << endl;
    }
  }



}

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testTrackAssociator);



