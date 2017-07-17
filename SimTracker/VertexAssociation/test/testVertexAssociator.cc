#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimTracker/VertexAssociation/test/testVertexAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/Math/interface/Vector.h"

#include <memory>
#include <iostream>
#include <string>

//class TrackAssociator;
class TrackAssociatorByHits;
class TrackerHitAssociator;

using namespace reco;
using namespace std;
using namespace edm;


testVertexAssociator::testVertexAssociator(edm::ParameterSet const& conf) :
  associatorByTracksToken(consumes<reco::VertexToTrackingVertexAssociator>(conf.getUntrackedParameter<edm::InputTag>("vertexAssociation")))
{

  vertexCollection_ = conf.getUntrackedParameter<edm::InputTag>( "vertexCollection" );

  n_event_ = 0;
  n_rs_vertices_ = 0;
  n_rs_vtxassocs_ = 0;
  n_sr_vertices_ = 0;
  n_sr_vtxassocs_ = 0;

}

testVertexAssociator::~testVertexAssociator() {


}

void testVertexAssociator::beginJob() {


  edm::Service<TFileService> fs; 
  
   //RecoToSim Histos

  rs_dist = fs->make<TH1F>("rs_dist","r Miss Distance (cm)",100,0,0.1);
  rs_recz = fs->make<TH1F>("rs_recz","z, Reconstructed Vertex (cm)", 200, -25.0,25.0);
  rs_simz = fs->make<TH1F>("rs_simz","z, Simulated Vertex (cm)",     200, -25.0,25.0);
  rs_nsimtrk = fs->make<TH1F>("rs_nsimtrk","# of tracks, Simulated",    501,-0.5,500.5);  
  rs_nrectrk = fs->make<TH1F>("rs_nrectrk","# of tracks, Reconstructed",501,-0.5,500.5);
  rs_qual = fs->make<TH1F>("rs_qual","Quality of Match",51,-0.01,1.01);
  rs_chi2norm = fs->make<TH1F>("rs_chi2norm","Vertex Normalized Chi2",100, 0, 10.);
  rs_chi2prob = fs->make<TH1F>("rs_chi2prob","Vertex Chi2 Probability",100, 0, 1.);
  rs_resx = fs->make<TH1F>("rs_resx","rec - sim Distance (cm)",100,-0.05,0.05);  
  rs_resy = fs->make<TH1F>("rs_resy","rec - sim Distance (cm)",100,-0.05,0.05);  
  rs_resz = fs->make<TH1F>("rs_resz","rec - sim Distance (cm)",100,-0.05,0.05);  
  rs_pullx = fs->make<TH1F>("rs_pullx","(rec - sim)/err_rec ",100,-10.,10.);  
  rs_pully = fs->make<TH1F>("rs_pully","(rec - sim)/err_rec ",100,-10.,10.); 
  rs_pullz = fs->make<TH1F>("rs_pullz","(rec - sim)/err_rec ",100,-10.,10.); 

  //SimToReco Histos

  sr_dist = fs->make<TH1F>("sr_dist","r Miss Distance (cm)",100,0,0.1);
  sr_recz = fs->make<TH1F>("sr_recz","z, Reconstructed Vertex (cm)", 200, -25.0,25.0);
  sr_simz = fs->make<TH1F>("sr_simz","z, Simulated Vertex (cm)",     200, -25.0,25.0);
  sr_nsimtrk = fs->make<TH1F>("sr_nsimtrk","# of tracks, Simulated",    501,-0.5,500.5);  
  sr_nrectrk = fs->make<TH1F>("sr_nrectrk","# of tracks, Reconstructed",501,-0.5,500.5);
  sr_qual = fs->make<TH1F>("sr_qual","Quality of Match",51,-0.01,1.01);
  sr_chi2norm = fs->make<TH1F>("sr_chi2norm","Vertex Normalized Chi2",100, 0, 10.);
  sr_chi2prob = fs->make<TH1F>("sr_chi2prob","Vertex Chi2 Probability",100, 0, 1.);
  sr_resx = fs->make<TH1F>("sr_resx","rec - sim Distance (cm)",100,-0.05,0.05);  
  sr_resy = fs->make<TH1F>("sr_resy","rec - sim Distance (cm)",100,-0.05,0.05);  
  sr_resz = fs->make<TH1F>("sr_resz","rec - sim Distance (cm)",100,-0.05,0.05);  
  sr_pullx = fs->make<TH1F>("sr_pullx","(rec - sim)/err_rec ",100,-10.,10.);  
  sr_pully = fs->make<TH1F>("sr_pully","(rec - sim)/err_rec ",100,-10.,10.); 
  sr_pullz = fs->make<TH1F>("sr_pullz","(rec - sim)/err_rec ",100,-10.,10.); 


}

void testVertexAssociator::endJob() {
 
  std::cout << std::endl;
  std::cout << " ====== Total Number of analyzed events: " << n_event_        << " ====== " << std::endl;
  std::cout << " ====== Total Number of R2S vertices:    " << n_rs_vertices_  << " ====== " << std::endl;
  std::cout << " ====== Total Number of R2S vtx assocs:  " << n_rs_vtxassocs_ << " ====== " << std::endl;
  std::cout << " ====== Total Number of S2R vertices:    " << n_sr_vertices_  << " ====== " << std::endl;
  std::cout << " ====== Total Number of S2R vtx assocs:  " << n_sr_vtxassocs_ << " ====== " << std::endl;
}

void testVertexAssociator::analyze(const edm::Event& event, const edm::EventSetup& setup)
{

  using namespace edm;
  using namespace reco;

  edm::ESHandle<MagneticField> theMF;
  setup.get<IdealMagneticFieldRecord>().get(theMF);

  edm::Handle<VertexToTrackingVertexAssociator> theTracksAssociator;
  event.getByToken(associatorByTracksToken, theTracksAssociator);
  associatorByTracks = theTracksAssociator.product();

  ++n_event_;

  std::cout << "*** Analyzing " << event.id() << " n_event = " << n_event_ << std::endl << std::endl;

  edm::Handle<TrackingVertexCollection>  TVCollection;
  event.getByLabel("mix","MergedTrackTruth",TVCollection);
  const TrackingVertexCollection tVC   = *(TVCollection.product());

  //Vertex Collection
  edm::Handle<edm::View<reco::Vertex> > vertexCollection;
  event.getByLabel(vertexCollection_, vertexCollection);
  const edm::View<reco::Vertex> vC = *(vertexCollection.product());

  cout << endl;
  cout << "                      ****************** Before Assocs ****************** " << endl << endl;  

  cout << "vertexCollection.size() = " << vC.size() << endl;
  cout << "TVCollection.size()     = " << tVC.size() << endl;
  
  cout << endl;
  cout << "                      ****************** Reco To Sim ****************** " << endl << endl;

  //cout << "-- Associator by hits --" << endl;
  reco::VertexRecoToSimCollection r2sVertices = associatorByTracks->associateRecoToSim(vertexCollection,TVCollection);

  reco::VertexSimToRecoCollection s2rVertices = associatorByTracks->associateSimToReco(vertexCollection,TVCollection);

  cout << endl;
  cout << "VertexRecoToSim size           = " <<  r2sVertices.size() << " ; VertexSimToReco size           = " << r2sVertices.size() << " " << endl;
 
 
  cout << endl << " [testVertexAssociator] Analyzing Reco To Sim" << endl;

  int cont_recvR2S = 0;

  for (reco::VertexRecoToSimCollection::const_iterator iR2S = r2sVertices.begin(); iR2S != r2sVertices.end(); ++iR2S, ++cont_recvR2S) {

    ++n_rs_vertices_;

    reco::VertexBaseRef recVertex = iR2S->key;
    math::XYZPoint recPos = recVertex->position();

    double nrectrk = recVertex->tracksSize();

    std::vector<std::pair<TrackingVertexRef, double> > simVertices = iR2S->val;

    int cont_simvR2S = 0;
    for (
        std::vector<std::pair<TrackingVertexRef, double> >::const_iterator iMatch = simVertices.begin();
        iMatch != simVertices.end(); 
        ++iMatch, ++cont_simvR2S
    ) {

      TrackingVertexRef simVertex =  iMatch->first;
      math::XYZTLorentzVectorD simVec = (iMatch->first)->position();
      math::XYZPoint simPos = math::XYZPoint(simVec.x(),simVec.y(),simVec.z());

      ++n_rs_vtxassocs_;

      cout << "rec vertex " << cont_recvR2S << " has associated sim vertex " << cont_simvR2S << endl;
 
      double nsimtrk = simVertex->daughterTracks().size();
      double qual  = iMatch->second;

      double chi2norm = recVertex->normalizedChi2();
      double chi2prob = ChiSquaredProbability( recVertex->chi2(), recVertex->ndof() ); 
        
      double resx = recVertex->x() - simVertex->position().x();
      double resy = recVertex->y() - simVertex->position().y();
      double resz = recVertex->z() - simVertex->position().z();	
      double pullx = (recVertex->x() - simVertex->position().x())/recVertex->xError();
      double pully = (recVertex->y() - simVertex->position().y())/recVertex->yError();
      double pullz = (recVertex->z() - simVertex->position().z())/recVertex->zError();
      double dist = sqrt(resx*resx+resy*resy+resz*resz);

      cout << "            R2S: recPos = " << recPos << " ; simPos = " << simPos << endl;

      rs_resx->Fill(resx);
      rs_resy->Fill(resy);
      rs_resz->Fill(resz);
      rs_pullx->Fill(pullx);
      rs_pully->Fill(pully);
      rs_pullz->Fill(pullz);
      rs_dist->Fill(dist);
      rs_simz->Fill(simPos.Z());
      rs_recz->Fill(recPos.Z());
      rs_nsimtrk->Fill(nsimtrk);
      rs_nrectrk->Fill(nrectrk);
      rs_qual->Fill(qual);
      rs_chi2norm->Fill(chi2norm);
      rs_chi2prob->Fill(chi2prob);	

    }//end simVertices
    
  }//end iR2S


  cout << endl << "                      ****************** Sim To Reco ****************** " << endl << endl;

  cout << endl << "[testVertexAssociator] Analyzing Sim To Reco" << endl;

  int cont_simvS2R = 0;
  for (reco::VertexSimToRecoCollection::const_iterator iS2R = s2rVertices.begin(); iS2R != s2rVertices.end(); ++iS2R, ++cont_simvS2R) {

    ++n_sr_vertices_;

    TrackingVertexRef simVertex = iS2R->key;
    math::XYZTLorentzVectorD simVec = simVertex->position();
    math::XYZPoint   simPos = math::XYZPoint(simVec.x(),simVec.y(),simVec.z());

    double nsimtrk = simVertex->daughterTracks().size();

    std::vector<std::pair<VertexBaseRef, double> > recoVertices = iS2R->val;

    int cont_recvS2R = 0;

    for (
      std::vector<std::pair<VertexBaseRef, double> >::const_iterator iMatch = recoVertices.begin();
      iMatch != recoVertices.end();
      ++iMatch, ++cont_recvS2R
     ) {

      VertexBaseRef recVertex = iMatch->first;
      math::XYZPoint recPos = recVertex->position();

      ++n_sr_vtxassocs_;

      cout << "sim vertex " << cont_simvS2R << " has associated rec vertex " << cont_recvS2R << endl;

      double nrectrk = recVertex->tracksSize();
      double qual  = iMatch->second;

      double chi2norm = recVertex->normalizedChi2();
      double chi2prob = ChiSquaredProbability( recVertex->chi2(), recVertex->ndof() ); 
         
      double resx = recVertex->x() - simVertex->position().x();
      double resy = recVertex->y() - simVertex->position().y(); 
      double resz = recVertex->z() - simVertex->position().z(); 	
      double pullx = ( recVertex->x() - simVertex->position().x() )/recVertex->xError();
      double pully = ( recVertex->y() - simVertex->position().y() )/recVertex->yError();
      double pullz = ( recVertex->z() - simVertex->position().z() )/recVertex->zError();
      double dist = sqrt(resx*resx+resy*resy+resz*resz);


      cout << "            S2R: simPos = " << simPos << " ; recPos = " << recPos << endl;

      sr_resx->Fill(resx);
      sr_resy->Fill(resy);
      sr_resz->Fill(resz);
      sr_pullx->Fill(pullx);
      sr_pully->Fill(pully);
      sr_pullz->Fill(pullz);
      sr_dist->Fill(dist);
      sr_simz->Fill(simPos.Z());
      sr_recz->Fill(recPos.Z());
      sr_nsimtrk->Fill(nsimtrk);
      sr_nrectrk->Fill(nrectrk);
      sr_qual->Fill(qual);
      sr_chi2norm->Fill(chi2norm);
      sr_chi2prob->Fill(chi2prob);	
  
    }//end recoVertices

  }//end iS2R

  cout << endl;

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testVertexAssociator);

