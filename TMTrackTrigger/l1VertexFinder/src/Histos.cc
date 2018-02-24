
#include "TMTrackTrigger/l1VertexFinder/interface/Histos.h"


#include <algorithm>
#include <array>
#include <fstream>
#include <unordered_set>

#include <TH1F.h>
#include <TH2F.h>
#include <TPad.h>
#include <TProfile.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TEfficiency.h>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TMTrackTrigger/l1VertexFinder/interface/InputData.h"
#include "TMTrackTrigger/l1VertexFinder/interface/L1fittedTrack.h"
#include "TMTrackTrigger/l1VertexFinder/interface/Settings.h"
#include "TMTrackTrigger/l1VertexFinder/interface/VertexFinder.h"




using namespace std;

namespace l1tVertexFinder {

//=== Book all histogram

void Histos::book() {
  TH1::SetDefaultSumw2(true);

  // Book histograms studying vertex reconstruction performance
  this->bookVertexReconstruction();
}


void Histos::bookVertexReconstruction(){
  TFileDirectory inputDir = fs_->mkdir("VertexReconstruction");

  hisGenVertexPt_ = inputDir.make<TH1F>("hisGenVertexPt_", "hisGenVertexPt_",100,0,500);
  hisGenTkVertexPt_ = inputDir.make<TH1F>("hisGenTkVertexPt_", "hisGenTkVertexPt_",100,0,500);
  hisGenVertexTrackPt_ = inputDir.make<TH1F>("hisGenVertexTrackPt_", "hisGenVertexTrackPt_",50,0,300);
  hisGenVertexNumTracks_ = inputDir.make<TH1F>("hisGenVertexNumTracks_", "hisGenVertexNumTracks_",20,0,50);
  hisGenVertexMET_ = inputDir.make<TH1F>("hisGenVertexMET_", "hisGenVertexMET_",100,0,500);
  hisGenTkVertexMET_ = inputDir.make<TH1F>("hisGenTkVertexMET_", "hisGenTkVertexMET_",100,0,500);

  hisPUVertexPt_ = inputDir.make<TH1F>("hisPUVertexPt_", "hisPUVertexPt_",50,0,300);
  hisPUVertexTrackPt_ = inputDir.make<TH1F>("hisPUVertexTrackPt_", "hisPUVertexTrackPt_",50,0,300);
  hisPUVertexNumTracks_ = inputDir.make<TH1F>("hisPUVertexNumTracks_", "hisPUVertexNumTracks_",20,0,50);

  hisRecoGenuineVertexVsGenTkVertexPt_ = inputDir.make<TH1F>("hisRecoGenuineVertexVsGenTkVertexPt_", "hisRecoGenuineVertexVsGenTkVertexPt_",100,0,500);
  hisRecoVertexVsGenVertexPt_ = inputDir.make<TH1F>("hisRecoVertexVsGenVertexPt_", "hisRecoVertexVsGenVertexPt_",100,0,500);
  hisRecoVertexVsGenTkVertexPtForEff_ = inputDir.make<TH1F>("hisRecoVertexVsGenTkVertexPtForEff_", "hisRecoVertexVsGenTkVertexPtForEff_",100,0,500);
  hisRecoVertexVsGenTkVertexMETForEff_ = inputDir.make<TH1F>("hisRecoVertexVsGenTkVertexMETForEff_", "hisRecoVertexVsGenTkVertexMETForEff_",100,0,500);
  hisRecoVertexVsNumGenTracks_ = inputDir.make<TH1F>("hisRecoVertexVsNumGenTracks_", "hisRecoVertexVsNumGenTracks_",20,0,50);
  hisRecoVertexVsGenMET_ = inputDir.make<TH1F>("hisRecoVertexVsGenMET_", "hisRecoVertexVsGenMET_",100,0,500);
  hisRecoGenuineVertexVsGenMET_ = inputDir.make<TH1F>("hisRecoGenuineVertexVsGenMET_", "hisRecoGenuineVertexVsGenMET_",100,0,500);
  hisRecoGenuineVertexVsGenTkMET_ = inputDir.make<TH1F>("hisRecoGenuineVertexVsGenTkMET_", "hisRecoGenuineVertexVsGenTkMET_",100,0,500);
  hisNoRecoVertices_                 = inputDir.make<TH1F>("hisNoRecoVertices_","No. reconstructed Vertices; No. reco vertices; Events",50,0,50);
  hisNoPileUpVertices_               = inputDir.make<TH1F>("hisNoPileUpVertices_","No. pile-up Vertices; No. pile-up vertices; Events",50,0,50);
  hisNoRecoVsNoTruePileUpVertices_   = inputDir.make<TH2F>("hisNoRecoVsNoTruePileUpVertices_","No. reconstructed pile-up vertices vs. no. true pile-up vertices; No. reco pile-up vertices; No. true pile-up vertices",50,0,50,50,0,50);
  hisRecoVertexZ0Resolution_         = inputDir.make<TH1F>("hisRecoVertexZ0Resolution","Reconstructed primary vertex z_{0} resolution; z_{0} Resolution [cm]; Counts", 100, 0., 1.);
  hisRecoVertexPTResolution_         = inputDir.make<TH1F>("hisRecoVertexPTResolution","Reconstructed primary vertex p_{T} relative resolution; p_{T} relative Resolution; Counts", 100, 0, 1.);
  hisRecoVertexPTResolutionVsTruePt_ = inputDir.make<TProfile>("hisRecoVertexPTResolutionVsTruePt","Reconstructed primary vertex relative p_{T} resolution vs. True Pt; True p_{T}; p_{T} Resolution [GeV]", 100, 0, 500);
  hisRecoVertexPTVsTruePt_ = inputDir.make<TH2F>("hisRecoVertexPtVsTruePt_","Reconstructed primary vertex p_{T}  vs. True Pt; p_{T} [GeV]; True p_{T}", 100, 0, 500.,100, 0, 500.);
  hisRecoVertexMETVsTrueMET_ = inputDir.make<TH2F>("hisRecoVertexMETVsTrueMET_","Reconstructed primary vertex MET vs. True MET; MET [GeV]; True MET", 100, 0, 500.,100, 0, 500.);
  hisNoTracksFromPrimaryVertex_      = inputDir.make<TH2F>("hisNoTracksFromPrimaryVertex_","No. of Tracks from Primary Vertex (Reco vs. Truth); no. Reco Tracks in PV; no. Truth Tracks in PV ",50,0,50,50,0,50);
  hisNoTrueTracksFromPrimaryVertex_  = inputDir.make<TH2F>("hisNoTrueTracksFromPrimaryVertex_","No. of Matched Tracks from Primary Vertex (Reco vs. Truth); no. Reco Tracks in PV; no. Truth Tracks in PV ",50,0,50,50,0,50);
  hisRecoPrimaryVertexZ0width_       = inputDir.make<TH1F>("hisRecoPrimaryVertexZ0width_", "Reconstructed primary vertex z_{0} width", 100, 0, 0.5);
  hisRatioMatchedTracksInPV_         = inputDir.make<TH1F>("hisRatioMatchedTracksInPV", "Primary vertex matching ratio ", 20, 0, 1.);
  hisFakeTracksRateInPV_             = inputDir.make<TH1F>("hisFakeTracksRateInPV", "Percentage of fake tracks in reconstructed primary vertex", 20, 0, 1.);
  hisTrueTracksRateInPV_             = inputDir.make<TH1F>("hisTrueTracksRateInPV", "Percentage of true tracks in reconstructed primary vertex", 20, 0, 1.);

  hisRecoPrimaryVertexVsTrueZ0_      = inputDir.make<TH1F>("hisRecoPrimaryVertexVsTrueZ0_","No. of reconstructed primary vertices per true z_{0}; true z_{0} [cm]; No. Vertices", 50, -25.,25.);
  hisRecoPrimaryVertexResolutionVsTrueZ0_      = inputDir.make<TProfile>("hisRecoPrimaryVertexResolutionVsTrueZ0_","No. of reconstructed primary vertices per true z_{0}; true z_{0} [cm]; Resolution [mm]", 100, -25.,25.);
  hisRecoVertexMET_ = inputDir.make<TH1F>("hisRecoVertexMET_","; L1TrkMET (GeV) ; Entries", 50, 0, 200);
  hisRecoVertexMETResolution_ = inputDir.make<TProfile>("hisRecoVertexResolutionMET_","; GenTrkMET (GeV) ; |L1TrkMET - GenMET|/GenMET", 50, 0, 200);
  hisRecoVertexPT_ = inputDir.make<TH1F>("hisRecoVertexPT_","; #Sigma_{vtx} p_{T} (GeV) ; Entries", 100, 0, 500);
  hisRecoPileUpVertexPT_ = inputDir.make<TH1F>("hisRecoPileUpVertexPT_","; #Sigma_{vtx} p_{T} (GeV) ; Entries", 50, 0, 200);
  hisRecoVertexOffPT_ = inputDir.make<TH1F>("hisRecoVertexOffPT_","; #Sigma_{vtx} p_{T} (GeV) ; Entries", 50, 0, 200);
  hisRecoVertexTrackRank_ = inputDir.make<TH1F>("hisRecoVertexTrackRank_","; Track Rank; Entries", 20, 0, 20);

  // Plot number of reconstructed vertices against number of true vertices
  // Plot number of reconstructed vertices against number of true vertices
  
  // *** Vertex Reconstruction algorithm Plots ***
  hisUnmatchedVertexZ0distance_            = inputDir.make<TH1F>("hisUnmatchedVertexZ0distance_"," Unmatched primary vertex z_{0} - true PV z_{0}; |z_{0}^{reco} - z_{0}^{true}| [cm]; Counts", 200, 1., 5.);
  hisUnmatchZ0distance_              = inputDir.make<TH1F>("hisUnmatchZ0distance_", "z0 distance from reconstructed privary vertex of unmatched tracks; |z_{0}^{track} - z_{0}^{vertex}|; no. L1 Tracks", 100, 0, 5.);
  hisUnmatchZ0MinDistance_              = inputDir.make<TH1F>("hisUnmatchZ0MinDistance_", "z0 distance from the closest track in reconstructed privary vertex of unmatched tracks; |z_{0}^{track} - z_{0}^{PV track}|; no. L1 Tracks", 100, 0, 5.);
  hisUnmatchPt_                      = inputDir.make<TH1F>("hisUnmatchPt_", "Transverse momentum of unmatched PV tracks; p_{T} [GeV/c]; no. L1 Tracks", 100, 0, 100.);
  hisUnmatchEta_                      = inputDir.make<TH1F>("hisUnmatchEta_", "#eta of unmatched PV tracks; #eta; no. L1 Tracks", 100, 2.4, 2.4);
  hisUnmatchTruePt_                      = inputDir.make<TH1F>("hisUnmatchTruePt_", "True transverse momentum of unmatched PV tracks; p_{T} [GeV/c]; no. L1 Tracks", 100, 0, 100.);
  hisUnmatchTrueEta_                      = inputDir.make<TH1F>("hisUnmatchTrueEta_", "True #eta of unmatched PV tracks; #eta; no. L1 Tracks", 100, 2.4, 2.4);
  hisUnmatchedPVtracks_              = inputDir.make<TH1F>("hisUnmatchedPVtracks_", " No. of tracks from primary collision that are misassigned; No. misassigned Tracks, No. Events ", 100, 0, 100);
  
  

  // *** TDR vertex reconstruction algorithm ***
  hisTDRVertexZ0Resolution_            = inputDir.make<TH1F>("hisTDRVertexZ0Resolution","TDR Reconstructed primary vertex z_{0} resolution; z_{0} Resolution [cm]; Counts", 100, 0., 1);
  hisTDRUnmatchedVertexZ0distance_            = inputDir.make<TH1F>("hisTDRUnmatchedVertexZ0distance_","TP Unmatched primary vertex z_{0} - true PV z_{0}; |z_{0}^{TP} - z_{0}^{true}| [cm]; Counts", 200, 1., 5.);

  hisTDRVertexPTResolution_            = inputDir.make<TH1F>("hisTDRVertexPTResolution","TDR Reconstructed primary vertex p_{T} relative resolution; p_{T} relative resolution; Counts", 100, 0, 1.);
  hisTDRVertexPTResolutionVsTruePt_    = inputDir.make<TProfile>("hisTDRVertexPTResolutionVsTruePt","TDR Reconstructed primary vertex relative p_{T} resolution vs. True Pt; True p_{T}; p_{T} Resolution [GeV]", 100, 0, 500);
  hisTDRVertexPTVsTruePt_              = inputDir.make<TH2F>("hisTDRVertexPtVsTruePt_","TDR Reconstructed primary vertex p_{T}  vs. True Pt; p_{T} [GeV]; True p_{T}", 100, 0, 500.,100, 0, 500.);
  hisTDRVertexMETVsTrueMET_            = inputDir.make<TH2F>("hisTDRVertexMETVsTrueMET_","TDR Reconstructed primary vertex MET  vs. True MET; MET [GeV]; True MET", 100, 0, 500.,100, 0, 500.);
  hisTDRNoTracksFromPrimaryVertex_     = inputDir.make<TH2F>("hisTDRNoTracksFromPrimaryVertex_","No. of Tracks from Primary Vertex (TDR vs. Truth); no. Reco Tracks in PV; no. Truth Tracks in PV ",50,0,50,50,0,50);
  hisTDRNoTrueTracksFromPrimaryVertex_ = inputDir.make<TH2F>("hisTDRNoTrueTracksFromPrimaryVertex_","No. of Matched Tracks from Primary Vertex (TDR vs. Truth); no. Reco Tracks in PV; no. Truth Tracks in PV ",50,0,50,50,0,50);
  hisTDRPrimaryVertexZ0width_          = inputDir.make<TH1F>("hisTDRPrimaryVertexZ0width_", "TDRReconstructed primary vertex z_{0} width", 100, 0, 0.5);
  hisRatioMatchedTracksInTDRPV_         = inputDir.make<TH1F>("hisRatioMatchedTracksInTDRPV", "TDR Primary vertex matching ratio ", 100, 0, 1.);
  hisFakeTracksRateInTDRPV_             = inputDir.make<TH1F>("hisFakeTracksRateInTDRPV", "Percentage of fake tracks in TDR recontructed primary vertex", 100, 0, 1.);
  hisTrueTracksRateInTDRPV_             = inputDir.make<TH1F>("hisTrueTracksRateInTDRPV", "Percentage of true tracks in TDR recontructed primary vertex", 100, 0, 1.);
  
  hisTDRUnmatchZ0distance_              = inputDir.make<TH1F>("hisTDRUnmatchZ0distance_", "z0 distance from TDR reconstructed privary vertex of unmatched tracks; |z_{0}^{track} - z_{0}^{vertex}|; no. L1 Tracks", 100, 0, 5.);
  hisTDRUnmatchZ0MinDistance_              = inputDir.make<TH1F>("hisTDRUnmatchZ0MinDistance_", "z0 distance from the closest track in TDR reconstructed privary vertex of unmatched tracks; |z_{0}^{track} - z_{0}^{PV track}|; no. L1 Tracks", 100, 0, 5.);
  hisTDRUnmatchPt_                      = inputDir.make<TH1F>("hisTDRUnmatchPt_", "Transverse momentum of unmatched TDR PV tracks; p_{T} [GeV/c]; no. L1 Tracks", 100, 0, 100.);
  hisTDRUnmatchEta_                      = inputDir.make<TH1F>("hisTDRUnmatchEta_", "#eta of unmatched TDR PV tracks; #eta; no. L1 Tracks", 100, 2.4, 2.4);
  hisTDRUnmatchTruePt_                      = inputDir.make<TH1F>("hisTDRUnmatchTruePt_", "True transverse momentum of unmatched TDR PV tracks; p_{T} [GeV/c]; no. L1 Tracks", 100, 0, 100.);
  hisTDRUnmatchTrueEta_                      = inputDir.make<TH1F>("hisTDRUnmatchTrueEta_", "True #eta of unmatched TDR PV tracks; #eta; no. L1 Tracks", 100, 2.4, 2.4);
  hisTDRUnmatchedPVtracks_              = inputDir.make<TH1F>("hisTDRUnmatchedPVtracks_", " No. of tracks from primary collision that are misassigned (TDR); No. misassigned Tracks, No. Events ", 100, 0, 100);
  hisTDRPrimaryVertexVsTrueZ0_      = inputDir.make<TH1F>("hisTDRPrimaryVertexVsTrueZ0_","No. of reconstructed primary vertices per true z_{0}; true z_{0} [cm]; No. Vertices", 50, -25.,25.);
  hisTDRPrimaryVertexResolutionVsTrueZ0_      = inputDir.make<TProfile>("hisTDRPrimaryVertexResolutionVsTrueZ0_","No. of reconstructed primary vertices per true z_{0} (Technical Proposal Algo); true z_{0} [cm]; Resolution [mm]", 100, -25.,25.);
  hisTDRVertexMET_ = inputDir.make<TH1F>("hisTDRVertexMET_","; L1TrkMET (GeV) ; Entries", 50, 0, 200);
  hisTDRVertexPT_ = inputDir.make<TH1F>("hisTDRVertexPT_","; #Sigma_{vtx} p_{T} (GeV) ; Entries", 50, 0, 200);
  hisTDRPileUpVertexPT_ = inputDir.make<TH1F>("hisTDRPileUpVertexPT_","; #Sigma_{vtx} p_{T} (GeV) ; Entries", 50, 0, 200);
  hisTDRVertexOffPT_ = inputDir.make<TH1F>("hisTDRVertexOffPT_","; #Sigma_{vtx} p_{T} (GeV) ; Entries", 50, 0, 200);


  // ** PileUp vertices plot **

  hisRecoPileUpVertexZ0width_        = inputDir.make<TH1F>("hisRecoPileUPVertexZ0width_", "Reconstructed pile-up vertex z_{0} width", 100, 0, 1.);
  hisRecoPileUpVertexZ0resolution_   = inputDir.make<TH1F>("hisRecoPileUPVertexZ0resolution_", "Reconstructed pile-up vertex z_{0} resolution; #sigma_{z_{0}} [cm]", 100, 0, 1.);
  
  hisRecoVertexZ0Spacing_            = inputDir.make<TH1F>("hisRecoVertexZ0Spacing", "Reconstructed intravertex z_{0} distance", 100, 0., 5.);
  
  hisPrimaryVertexTrueZ0_            = inputDir.make<TH1F>("hisPrimaryVertexTrueZ0_","No. of gen primary vertices per true z_{0}; true z_{0} [cm]; No. Vertices", 50, -25.,25.);

  hisPrimaryVertexZ0width_           = inputDir.make<TH1F>("hisPrimaryVertexZ0width_", "Primary vertex z_{0} width", 100, 0, 0.5);
  hisPileUpVertexZ0_                 = inputDir.make<TH1F>("hisPileUpVertexZ0", "Pile Up vertex z_{0} position", 200, -15., 15.);
  hisPileUpVertexZ0Spacing_          = inputDir.make<TH1F>("hisPileUpVertexZ0Spacing", "Pile Up intravertex z_{0} distance", 100, 0., 5.);
  hisPileUpVertexZ0width_            = inputDir.make<TH1F>("hisPileUpVertexZ0width_", "Pile Up vertex z_{0} width", 100, 0, 0.5);
  
  hisLostPVtracks_                   = inputDir.make<TH1F>("hisLostPVtracks_", " No. of tracks from primary collision that are not found by the L1 Track Finder; No. Lost Tracks, No. Events ", 100, 0, 100);

  hisTrkMETvsGenMET_                 = inputDir.make<TProfile>("hisTrkMETvsGenMET_"," Generated L1 Track MET vs Generated MET ", 20, 0, 500);
  hisRecoTrkMETvsGenMET_             = inputDir.make<TProfile>("hisRecoTrkMETvsGenMET_"," Reconstructed L1 Track MET vs Generated MET ", 20, 0, 500);
  hisTDRTrkMETvsGenMET_              = inputDir.make<TProfile>("hisTDRTrkMETvsGenMET_"," Technical Proposal L1 Track MET vs Generated MET ", 20, 0, 500);

  hisNumVxIterations_                = inputDir.make<TH1F>("hisNumVxIterations_","Number of Iterations (Vertex Reconstruction); No. Iterations; Entries",100,0,500);
  hisNumVxIterationsPerTrack_        = inputDir.make<TH1F>("hisNumVxIterationsPerTrack_","Number of Iterations per Track(Vertex Reconstruction); No. Iterations; Entries",100,0,200);
  
  hisCorrelatorInputTracks_             = inputDir.make<TH1F>("hisCorrelatorInputTracks_","Number of Input Tracks at L1 correlator; No. L1 Tracks; Entries", 30, 0, 200);
  hisCorrelatorTPInputTracks_             = inputDir.make<TH1F>("hisCorrelatorTPInputTracks_","Number of Input Tracks at L1 correlator; No. L1 Tracks; Entries", 30, 0, 200);

  hisCorrelatorInputVertices_             = inputDir.make<TH1F>("hisCorrelatorInputVertices_","Number of Input Vertices at L1 correlator; No. L1TkVertices; Entries", 25, 0, 100);
  hisCorrelatorTPInputVertices_             = inputDir.make<TH1F>("hisCorrelatorTPInputVertices_","Number of Input Vertices at L1 correlator; No. L1TKVertices; Entries", 25, 0, 100);

  grMET_.resize(4);
  grMET_tdr_.resize(4);
  hisMETevents_.resize(4);
  hisPTevents_.resize(4);
  hisRecoVertexVsGenTkMET_.resize(4);
  hisRecoVertexVsGenTkVertexPt_.resize(4);
  float genMet[4] = {50, 100, 200, 300};

  for(unsigned int i = 0; i < 4;  ++i ){
    grMET_[i] = inputDir.make<TGraphErrors>(10);
    grMET_tdr_[i] = inputDir.make<TGraphErrors>(10);
    ostringstream title, name;

    name << "hisRecoVertexVsGenTkMET_" << genMet[i];

    hisRecoVertexVsGenTkMET_[i] = inputDir.make<TH1F>(name.str().c_str(), name.str().c_str(),100,0,500);

    name.clear();
    name.str("");
    name << "hisRecoVertexVsGenTkVertexPt_" << genMet[i];
    hisRecoVertexVsGenTkVertexPt_[i] = inputDir.make<TH1F>(name.str().c_str(), name.str().c_str(),100,0,500);

    title << "Signal Efficiency vs. Bkg. rejection ( GenMET > "<< genMet[i] << " GeV); Signal Efficiency; Bkg Rejection power";
    name << "grSignVsBkgEffGenMET"<< genMet[i];

    grMET_[i]->SetTitle(title.str().c_str());
    grMET_[i]->SetName(name.str().c_str());
    grMET_[i]->SetMarkerStyle(21);
    grMET_[i]->GetXaxis()->SetRangeUser(0,1.1);
    grMET_[i]->GetYaxis()->SetRangeUser(0,1.1);
    // grMET_[i]->SetMarkerSize()

    title.clear();
    title.str("");
    title << "Signal Efficiency vs. Bkg. rejection ( GenMET > "<< genMet[i] << " GeV, Technical Proposal Algo); Signal Efficiency; Bkg Rejection power";
    name.clear();
    name.str("");
    name << "grSignVsBkgEffGenMET"<< genMet[i]<<"_TP";

    grMET_tdr_[i]->SetTitle(title.str().c_str());
    grMET_tdr_[i]->SetName(name.str().c_str());
    grMET_tdr_[i]->SetMarkerStyle(21);
    grMET_tdr_[i]->GetXaxis()->SetRangeUser(0,1.1);
    grMET_tdr_[i]->GetYaxis()->SetRangeUser(0,1.1);

    name.clear();
    name.str("");
    name << "hisMET"<< genMet[i];    
    hisMETevents_[i] = inputDir.make<TH1F>(name.str().c_str(), name.str().c_str(), 42, 0 ,42);

    name.clear();
    name.str("");
    name << "hisPT"<< genMet[i];    
    hisPTevents_[i] = inputDir.make<TH1F>(name.str().c_str(), name.str().c_str(), 22, 0 , 22);

  }

  noSignalEvents.assign(4,0);
  noBackgroundEvents.assign(4,0);
  noSignalEventsTDR.assign(4,0);
  noBackgroundEventsTDR.assign(4,0);

  std::vector<unsigned int> emptyVector;
  emptyVector.assign(10,0);

  noRecoSignalEvents.assign(4,emptyVector);
  noRecoBackgroundEvents.assign(4,emptyVector);
  noTDRSignalEvents.assign(4,emptyVector);
  noTDRBackgroundEvents.assign(4,emptyVector);

}


void Histos::fillVertexReconstruction(const InputData& inputData, const VertexFinder& vf, std::vector<L1fittedTrack> l1Tracks){
  cout << "Input Tracks to L1 Correlator " << vf.numInputTracks() << endl;

  // noEvents++;
  const Vertex&     TruePrimaryVertex = inputData.getPrimaryVertex();

  // create a map for associating fat reco tracks with their underlying
  // TTTrack pointers
  // std::map <const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>, L1fittedTrack> trackAssociationMap;
  std::map <const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>, const L1fittedTrack *> trackAssociationMap;

  // unsigned int index = 0;
  // get a list of reconstructed tracks with references to their TPs
  for (const auto & trackIt: l1Tracks) {
    trackAssociationMap.insert(std::pair<const edm::Ptr<TTTrack< Ref_Phase2TrackerDigi_ >>, const L1fittedTrack *>(trackIt.getTTTrackPtr(), &trackIt));
  }

  // Associate true primary vertex with the closest reconstructed vertex
  RecoVertex RecoPrimaryVertexBase = vf.PrimaryVertex();
  RecoVertex TDRVertexBase         = vf.TDRPrimaryVertex();

  RecoVertexWithTP * RecoPrimaryVertex = new RecoVertexWithTP(RecoPrimaryVertexBase, trackAssociationMap);
  RecoVertexWithTP * TDRVertex = new RecoVertexWithTP(TDRVertexBase, trackAssociationMap);

  RecoPrimaryVertex->computeParameters(settings_->vx_weightedmean());
  TDRVertex->computeParameters(settings_->vx_weightedmean());

  // update the primary vertex z0 if the algorithm is HPV, kmeans
  if (settings_->vx_algoId() == 6 || settings_->vx_algoId() == 5)
    RecoPrimaryVertex->setZ(RecoPrimaryVertexBase.z0());
  TDRVertex->setZ(TDRVertexBase.z0());

  hisGenVertexPt_->Fill(inputData.GenPt());
  hisGenTkVertexPt_->Fill(TruePrimaryVertex.pT());
  hisGenVertexMET_->Fill(inputData.GenMET());
  hisGenTkVertexMET_->Fill(TruePrimaryVertex.met());
  hisGenVertexNumTracks_->Fill(TruePrimaryVertex.numTracks());

  for(const TP& tp : TruePrimaryVertex.tracks()){
    hisGenVertexTrackPt_->Fill(tp.pt());
  }

  for(const Vertex& vertex : inputData.getPileUpVertices()){
    hisPUVertexPt_->Fill(vertex.pT());
    hisPUVertexNumTracks_->Fill(vertex.numTracks());
    for(const TP& tp : vertex.tracks()){
      hisPUVertexTrackPt_->Fill(tp.pt());
    }
  }

  if(RecoPrimaryVertex->pT() > 100.){
    hisRecoVertexVsGenVertexPt_->Fill(inputData.GenPt());
  }

  if(RecoPrimaryVertex->met() > 50.){
    hisRecoVertexVsGenMET_->Fill(inputData.GenMET());
  }

  if(settings_->debug()==7){
    cout << "** RECO VERTICES ***" << endl;
    for(RecoVertex vertex : vf.Vertices()){
      cout << "recovertex z0 "<< vertex.z0() << " pt "<< vertex.pT() << " highpt " << vertex.hasHighPt() << " numtracks "<< vertex.numTracks() << " numTrueTracks "<< vertex.numTrueTracks() << endl;
    }
 
    cout << "True PrimaryVertex z0 "<< TruePrimaryVertex.z0() << " pT "<< TruePrimaryVertex.pT() << " met "<< TruePrimaryVertex.met() << endl;
    cout << "Reco PrimaryVertex z0 "<< RecoPrimaryVertex->z0() << " pT "<< RecoPrimaryVertex->pT() << " met "<< RecoPrimaryVertex->met() << " nTracks "<< RecoPrimaryVertex->numTracks() << endl;
    cout << "TP PrimaryVertex z0 "<< TDRVertex->z0() << " pT "<< TDRVertex->pT() << " met "<< RecoPrimaryVertex->met() << endl;
  }

  unsigned int TrackRank = 0;
  for(unsigned int id = 0; id < vf.numVertices() ; ++id){
    if(id!=vf.PrimaryVertexId()){
      // this will not work since numTrueTracks() will return 0
      // to fix this, one needs to use reference to RecoVertex from vf.Vertices()
      // and use that to create a RecoVertexWithTP object that will have TP
      // specific information
      if(vf.Vertices()[id].numTrueTracks() > RecoPrimaryVertex->numTrueTracks()) TrackRank++;
    }
  }

  hisRecoVertexTrackRank_->Fill(TrackRank);

  hisCorrelatorInputTracks_->Fill(vf.numInputTracks());
  hisCorrelatorTPInputTracks_->Fill(vf.numInputTracks());

  hisNumVxIterations_->Fill(vf.NumIterations());
  hisNumVxIterationsPerTrack_->Fill(vf.IterationsPerTrack());
  

  hisTrkMETvsGenMET_->Fill(inputData.GenMET(), TruePrimaryVertex.met() );
  hisRecoTrkMETvsGenMET_->Fill(inputData.GenMET(), RecoPrimaryVertex->met());
  hisTDRTrkMETvsGenMET_->Fill(inputData.GenMET(), TDRVertex->met());

  hisNoRecoVertices_->Fill(vf.numVertices());
  hisNoPileUpVertices_->Fill(inputData.getRecoPileUpVertices().size());
  hisNoRecoVsNoTruePileUpVertices_->Fill(vf.numVertices(),inputData.getRecoPileUpVertices().size() );
  
  if(TruePrimaryVertex.numTracks() > 0) hisPrimaryVertexTrueZ0_->Fill(TruePrimaryVertex.z0());

  float z0res = TruePrimaryVertex.z0() - RecoPrimaryVertex->z0();
  float pTres = fabs(TruePrimaryVertex.pT() - RecoPrimaryVertex->pT());
  hisRecoVertexZ0Resolution_->Fill(fabs(z0res));

  // Vertex has been found
  if(fabs(z0res) < settings_->vx_resolution()) {
    float genMet[4] = {50, 100, 200, 300};
    for(unsigned int i = 0; i< 4 ;++i){
      if(RecoPrimaryVertex->met()>genMet[i])
        hisRecoVertexVsGenTkMET_[i]->Fill(TruePrimaryVertex.met());
      if(RecoPrimaryVertex->pT()>genMet[i]){
        hisRecoVertexVsGenTkVertexPt_[i]->Fill(TruePrimaryVertex.pT());
      }
    

      float minThreshold = genMet[i] - 0.5*genMet[i];
      float step = genMet[i]/10.;
      if(TruePrimaryVertex.pT() > genMet[i]){
        hisPTevents_[i]->Fill(0.5); 
        for(unsigned int j = 0 ; j < 10 ; ++j){
          float cut = minThreshold + step*j;
          if(RecoPrimaryVertex->pT() > cut) hisPTevents_[i]->Fill(1.5+j);
        }
      } else{
        hisPTevents_[i]->Fill(11.5);
        for(unsigned int j = 0 ; j < 10 ; ++j){
          float cut = minThreshold + step*j;
          if(RecoPrimaryVertex->pT() < cut) hisPTevents_[i]->Fill(12.5+j);
        }
      }

    }

    float METres = fabs(RecoPrimaryVertex->met() - TruePrimaryVertex.met())/TruePrimaryVertex.met();

    if(settings_->debug() == 7 and METres > 0.2){
      cout << "** RECO TRACKS in PV**" << endl;
      for(const L1fittedTrack* track : RecoPrimaryVertex->tracks() ){
        if(track->getMatchedTP() != nullptr) cout << "matched TP "<< track->getMatchedTP()->index() ;
        cout << " pT "<< track->pt() << " phi0 "<< track->phi0() << " z0 "<< track->z0() << endl;
      }
      cout << "** TRUE TRACKS in PV**" << endl;
      for(TP track: TruePrimaryVertex.tracks()){
        cout << "index " << track.index() <<" pT "<< track.pt() << " phi0 "<< track.phi0() << " z0 "<< track.z0() << " status "<< track.physicsCollision() << endl;
      }
    }


    hisRecoVertexMETResolution_->Fill(TruePrimaryVertex.met(), METres);

    if(RecoPrimaryVertex->pT() > 100.){
      hisRecoGenuineVertexVsGenTkVertexPt_->Fill(TruePrimaryVertex.pT());
    }

    if(RecoPrimaryVertex->met() > 50.){
      hisRecoGenuineVertexVsGenMET_->Fill(inputData.GenMET());
      hisRecoGenuineVertexVsGenTkMET_->Fill(TruePrimaryVertex.met());
    }

    hisRecoVertexVsNumGenTracks_->Fill(TruePrimaryVertex.numTracks());
    hisRecoVertexVsGenTkVertexPtForEff_->Fill(TruePrimaryVertex.pT());
    hisRecoVertexVsGenTkVertexMETForEff_->Fill(TruePrimaryVertex.met());
    hisRecoPrimaryVertexVsTrueZ0_->Fill(TruePrimaryVertex.z0());
  
    // ** Reconstructed Primary Vertex Histos **
    hisRecoVertexPTResolution_->Fill(pTres/TruePrimaryVertex.pT());
    hisRecoVertexPTResolutionVsTruePt_->Fill(TruePrimaryVertex.pT(), pTres/TruePrimaryVertex.pT() );
  
    hisRecoVertexPTVsTruePt_->Fill(RecoPrimaryVertex->pT(), TruePrimaryVertex.pT());
    hisRecoVertexMETVsTrueMET_->Fill(RecoPrimaryVertex->met(), TruePrimaryVertex.met());
    hisRecoVertexMET_->Fill(RecoPrimaryVertex->met());
    hisNoTracksFromPrimaryVertex_->Fill(RecoPrimaryVertex->numTracks(),TruePrimaryVertex.numTracks());
    hisNoTrueTracksFromPrimaryVertex_->Fill(RecoPrimaryVertex->numTrueTracks(),TruePrimaryVertex.numTracks());
    hisRecoPrimaryVertexZ0width_->Fill(RecoPrimaryVertex->z0width());
    hisRecoVertexPT_->Fill(RecoPrimaryVertex->pT());
    
    float matchratio = float(RecoPrimaryVertex->numTrueTracks())/float(TruePrimaryVertex.numTracks());
    if(matchratio > 1.) matchratio = 1.;
    hisRatioMatchedTracksInPV_->Fill(matchratio);
    float trueRate = float(RecoPrimaryVertex->numTrueTracks())/float(RecoPrimaryVertex->numTracks());
    hisTrueTracksRateInPV_->Fill(trueRate);
    float fakeRate = float(RecoPrimaryVertex->numTracks()-RecoPrimaryVertex->numTrueTracks())/float(RecoPrimaryVertex->numTracks());
    hisFakeTracksRateInPV_->Fill(fakeRate);
    hisRecoPrimaryVertexResolutionVsTrueZ0_->Fill(TruePrimaryVertex.z0(),fabs(z0res));

    for(unsigned int i = 0; i < 4; ++i ){
      float genmet = genMet[i];
      float met_steps = (genmet-genmet*0.3)/10;
      bool signal = false;
      if(TruePrimaryVertex.met() > genmet){
        noSignalEvents[i]++;
        signal = true;
      }
      else {
        noBackgroundEvents[i]++;
      }

      for(unsigned int j = 0; j < 10; ++j ){
        float cutmet = genmet*0.3 + j*met_steps;

        if(RecoPrimaryVertex->met() > cutmet){
          if(signal) noRecoSignalEvents[i][j]++;
        } else if(!signal){
          noRecoBackgroundEvents[i][j]++;
        }
      }
    }

  }
  else{
    

    hisRecoVertexOffPT_->Fill(RecoPrimaryVertex->pT());
    hisUnmatchedVertexZ0distance_->Fill(fabs(z0res));
    if(settings_->debug() == 7) {
      cout << "Vertex Reconstruction Algorithm doesn't find the correct the primary vertex (Delta Z = " << fabs(z0res) << ")"<<endl;
      
    }
  }

  if(settings_->debug() == 7){
    for(const L1fittedTrack* l1track : RecoPrimaryVertex->tracks()){
      if(l1track->getMatchedTP() == nullptr){
        cout << "FAKE track assigned to PV. Track z0: "<< l1track->z0() << " track pT "<< l1track->pt() << " chi2/ndof " << l1track->chi2dof() << " numstubs "<< l1track->getNumStubs() << endl;
      } else if(l1track->getMatchedTP()->physicsCollision() == 0){
        cout << "Pile-Up track assigned to PV. Track z0: "<< l1track->z0() << " track pT "<< l1track->pt() << endl;
      } else{
          cout << "Physics Collision track assigned to PV. Track z0: "<< l1track->z0() << " track pT "<< l1track->pt() << " numstubs "<< l1track->getNumStubs();
          cout << " (real values) id: "<< l1track->getMatchedTP()->index()<< " pT " <<  l1track->getMatchedTP()->pt() << " eta " << l1track->getMatchedTP()->eta() << " d0 "<< l1track->getMatchedTP()->d0() << " z0 "<< l1track->getMatchedTP()->z0() << " physicsCollision " << l1track->getMatchedTP()->physicsCollision() << " useForEff() " << l1track->getMatchedTP()->useForEff() << " pdg "<< l1track->getMatchedTP()->pdgId() << " tip "<< l1track->getMatchedTP()->tip() << endl;
      }
    }
  }


  // ** Technical Proposal algorithm
  float z0res_tdr = (TruePrimaryVertex.z0() - TDRVertex->z0());
  float pTres_tdr = fabs(TruePrimaryVertex.pT() - TDRVertex->pT());
  
  hisTDRVertexZ0Resolution_->Fill(fabs(z0res_tdr));

  if(fabs(z0res_tdr) < settings_->vx_resolution()){
    hisTDRPrimaryVertexVsTrueZ0_->Fill(TruePrimaryVertex.z0());
    hisTDRPrimaryVertexResolutionVsTrueZ0_->Fill(TruePrimaryVertex.z0(),fabs(z0res_tdr));
    hisTDRVertexPT_->Fill(TDRVertex->pT());
    hisTDRVertexMET_->Fill(TDRVertex->met());
    hisTDRVertexPTResolution_->Fill(pTres_tdr/TruePrimaryVertex.pT());
    hisTDRVertexPTResolutionVsTruePt_->Fill(TruePrimaryVertex.pT(), pTres_tdr/TruePrimaryVertex.pT() );
    hisTDRVertexPTVsTruePt_->Fill(TDRVertex->pT(), TruePrimaryVertex.pT());
    hisTDRVertexMETVsTrueMET_->Fill(TDRVertex->met(), TruePrimaryVertex.met());
    hisTDRNoTracksFromPrimaryVertex_->Fill(TDRVertex->numTracks(),TruePrimaryVertex.numTracks());
    hisTDRNoTrueTracksFromPrimaryVertex_->Fill(TDRVertex->numTrueTracks(),TruePrimaryVertex.numTracks());
    hisTDRPrimaryVertexZ0width_->Fill(TDRVertex->z0width());
  
    float matchratio_res = float(TDRVertex->numTrueTracks())/float(TruePrimaryVertex.numTracks());
    hisRatioMatchedTracksInTDRPV_->Fill(matchratio_res);
    float trueRate_res = float(TDRVertex->numTrueTracks())/float(TDRVertex->numTracks());
    hisTrueTracksRateInTDRPV_->Fill(trueRate_res);
    float fakeRate_res = float(TDRVertex->numTracks()-TDRVertex->numTrueTracks())/float(TDRVertex->numTracks());
    hisFakeTracksRateInTDRPV_->Fill(fakeRate_res);

    for(unsigned int i = 0; i < 3; ++i ){
      float genmet = 25. + i*25.;
      float met_steps = (genmet-10.)/10;
      bool signal = false;
      if(TruePrimaryVertex.met() > genmet){
        noSignalEventsTDR[i]++;
        signal = true;
      }
      else {
        noBackgroundEventsTDR[i]++;
      }

      for(unsigned int j = 0; j < 10; ++j ){
        float cutmet = 10. + j*met_steps;

        if(TDRVertex->met() > cutmet){
          if(signal) noTDRSignalEvents[i][j]++;
        } else if(!signal){
          noTDRBackgroundEvents[i][j]++;
        }
      }
    }

  } else{
    hisTDRVertexOffPT_->Fill(TDRVertex->pT());
    hisTDRUnmatchedVertexZ0distance_->Fill(fabs(z0res_tdr));
  }

  unsigned int lostTracks = 0;
  unsigned int misassignedTracks = 0;
  unsigned int misassignedTracks_tdr = 0;


  if(settings_->debug() == 7) cout << "*** Misassigned primary vertex tracks ***"<< endl;
  for(const TP& tp : TruePrimaryVertex.tracks()){
    bool found = false;
    // cout << tp.index() << " "<< endl;
    for(const L1fittedTrack* l1track : RecoPrimaryVertex->tracks()){
      if(l1track->getMatchedTP()!= nullptr){
        if(tp.index() == l1track->getMatchedTP()->index() ) {
          found = true;
          break;
        }
      }
    }

    if(!found){
      bool TrackIsReconstructed = false;
      for(const L1fittedTrackBase* l1trackIt: vf.FitTracks()){
        const L1fittedTrack * l1track = trackAssociationMap[l1trackIt->getTTTrackPtr()];
        if(l1track->getMatchedTP()!= nullptr){
          if(tp.index() == l1track->getMatchedTP()->index() ){
            TrackIsReconstructed = true;
            hisUnmatchZ0distance_->Fill(fabs(l1track->z0()-RecoPrimaryVertex->z0()));
            hisUnmatchPt_->Fill(l1track->pt());
            hisUnmatchEta_->Fill(l1track->eta());
            hisUnmatchTruePt_->Fill(tp.pt());
            hisUnmatchTrueEta_->Fill(tp.eta());

            double mindistance = 999.;
            for(const L1fittedTrack* vertexTrack : RecoPrimaryVertex->tracks()){
              if( fabs(vertexTrack->z0()-l1track->z0()) < mindistance ) mindistance = fabs(vertexTrack->z0()-l1track->z0());
            }
            hisUnmatchZ0MinDistance_->Fill(mindistance);

            if(settings_->debug()>5){
              cout << "PV Track assigned to wrong vertex. Track z0: "<< l1track->z0() << " PV z0: "<< RecoPrimaryVertex->z0() << " tp z0 "<< tp.z0() << " track pT "<< l1track->pt() << " tp pT "<< tp.pt() << " tp d0 "<< tp.d0() << " track eta "<< l1track->eta() << endl;
            }
            break;
          }
        }
      }

      if(!TrackIsReconstructed){
        lostTracks++;
      } else{
        misassignedTracks++;
      }
    }

    found = false;

    for(const L1fittedTrackBase* l1trackIt : TDRVertex->tracks()){
      const L1fittedTrack * l1track = trackAssociationMap[l1trackIt->getTTTrackPtr()];
      if(l1track->getMatchedTP()!= nullptr){
        // cout << l1track->getMatchedTP()->index() << " ";
        if(tp.index() == l1track->getMatchedTP()->index() ) {
          found = true;
          break;
        }
      }
    }

    if(!found){
      for(const L1fittedTrackBase* l1trackIt: vf.FitTracks()){
	const L1fittedTrack * l1track = trackAssociationMap[l1trackIt->getTTTrackPtr()];
        if(l1track->getMatchedTP()!= nullptr){
          if(tp.index() == l1track->getMatchedTP()->index() ){
            hisTDRUnmatchZ0distance_->Fill(fabs(l1track->z0()-TDRVertex->z0()));
            hisTDRUnmatchPt_->Fill(l1track->pt());
            hisTDRUnmatchEta_->Fill(l1track->eta());
            hisTDRUnmatchTruePt_->Fill(tp.pt());
            hisTDRUnmatchTrueEta_->Fill(tp.eta());
            misassignedTracks_tdr++;
            double mindistance = 999.;
            for(const L1fittedTrackBase* vertexTrack : TDRVertex->tracks()){
              if( fabs(vertexTrack->z0()-l1track->z0()) < mindistance ) mindistance = fabs(vertexTrack->z0()-l1track->z0());
            }
            hisTDRUnmatchZ0MinDistance_->Fill(mindistance);

            break;
          }
        }
      }
    }


  }

  hisLostPVtracks_->Fill(lostTracks);
  hisUnmatchedPVtracks_->Fill(misassignedTracks);
  hisTDRUnmatchedPVtracks_->Fill(misassignedTracks_tdr);

  hisPrimaryVertexZ0width_->Fill(TruePrimaryVertex.z0width());

  float z0distance = 0.;

  for(unsigned int i = 0; i<vf.Vertices().size(); ++i){
    if(i < vf.Vertices().size()-1){
      z0distance = vf.Vertices()[i+1].z0() - vf.Vertices()[i].z0();
      hisRecoVertexZ0Spacing_->Fill(z0distance);
    }
    if(i != vf.PrimaryVertexId()  ){ 
      hisRecoPileUpVertexZ0width_->Fill(vf.Vertices()[i].z0width());
      hisRecoPileUpVertexPT_->Fill(vf.Vertices()[i].pT());
      double PUres = 999.;
      for(unsigned int j = 0; j<inputData.getRecoPileUpVertices().size(); ++j){
        if(fabs(vf.Vertices()[i].z0()-inputData.getRecoPileUpVertices()[j].z0()) < PUres){
          PUres = fabs(vf.Vertices()[i].z0()-inputData.getRecoPileUpVertices()[j].z0());
        }
      }
      hisRecoPileUpVertexZ0resolution_->Fill(PUres);
    }
  }

  for(unsigned int i = 0; i<inputData.getRecoPileUpVertices().size(); ++i){
    if(i < inputData.getRecoPileUpVertices().size()-1){
      z0distance = inputData.getRecoPileUpVertices()[i+1].z0() - inputData.getRecoPileUpVertices()[i].z0();
      hisPileUpVertexZ0Spacing_->Fill(z0distance);
    }
    hisPileUpVertexZ0_->Fill(inputData.getRecoPileUpVertices()[i].z0());
    hisPileUpVertexZ0width_->Fill(inputData.getRecoPileUpVertices()[i].z0width());
  }


  if(settings_->debug()==7) cout << "================ End of Event =============="<< endl;

  delete RecoPrimaryVertex;
  delete TDRVertex;

}

//=== Book efficiency & fake rate histograms and print track-finding performance summary.

void Histos::endJobAnalysis() {
 
  // Vertex Efficiency
  TFileDirectory inputDir= fs_->mkdir("VertexEfficiency");

  PVefficiencyVsTrueZ0_ = inputDir.make<TEfficiency>(*hisRecoPrimaryVertexVsTrueZ0_,*hisPrimaryVertexTrueZ0_);
  PVefficiencyVsTrueZ0_->SetNameTitle("PVefficiencyVsTrueZ0_", "Primary Vertex Finding Efficiency; true z_{0}; Efficiency");

  tdrPVefficiencyVsTrueZ0_ = inputDir.make<TEfficiency>(*hisTDRPrimaryVertexVsTrueZ0_,*hisPrimaryVertexTrueZ0_);
  tdrPVefficiencyVsTrueZ0_->SetNameTitle("tdrPVefficiencyVsTrueZ0_", "Primary Vertex Finding Efficiency (Technical Proposal Algo); true z_{0}; Efficiency");





  cout << "==================== VERTEX RECONSTRUCTION ======================" << endl;
  
  if(settings_->vx_algoId()==0) cout << "GAP ALGORITHM with vertex gap = "<< settings_->vx_distance() << " cm "<< endl;
  if(settings_->vx_algoId()==1) cout << "SIMPLE MERGE CLUSTERING ALGORITHM with mininal vertex distance = "<< settings_->vx_distance() << " cm "<< endl;
  if(settings_->vx_algoId()==2) cout << "DBSCAN ALGORITHM with vertex gap = "<< settings_->vx_distance() << " cm "<< endl;
  if(settings_->vx_algoId()==3) cout << "PVR ALGORITHM with vertex gap = "<< settings_->vx_distance() << " cm "<< endl;
  if(settings_->vx_algoId()==4) cout << "AVR ALGORITHM with vertex gap = "<< settings_->vx_distance() << " cm "<< endl;

  cout << "Average no. Reconstructed Vertices: "<< hisNoRecoVertices_->GetMean() << "(" << hisNoRecoVertices_->GetMean()*100./(hisNoPileUpVertices_->GetMean()+1.)<<"%)"<< endl;
  cout << "Average ratio of matched tracks in primary vertex "<< hisRatioMatchedTracksInPV_->GetMean()*100 <<"%, Technical Proposal Algo : "<< hisRatioMatchedTracksInTDRPV_->GetMean()*100 << " % "<< endl;
  cout << "Averate ratio of fake tracks in primary vertex "<< hisFakeTracksRateInPV_->GetMean()*100 << "%, Technical Proposal Algo : "<< hisFakeTracksRateInTDRPV_->GetMean()*100 << " % "<< endl;
  cout << "Average PV z0 resolution "<< hisRecoVertexZ0Resolution_->GetMean() << " cm , Technical Proposal Algo : "<< hisTDRVertexZ0Resolution_->GetMean() << " cm "<< endl;

  float recoPVeff = double(hisRecoPrimaryVertexVsTrueZ0_->GetEntries())/double(hisPrimaryVertexTrueZ0_->GetEntries());
  float numRecoPV = double(hisRecoPrimaryVertexVsTrueZ0_->GetEntries());
  float numPVs = double(hisPrimaryVertexTrueZ0_->GetEntries());
  float numTDRPV = double(hisTDRPrimaryVertexVsTrueZ0_->GetEntries());

  float recoPVeff_err = sqrt((numRecoPV+1)*(numRecoPV+2)/((numPVs+2)*(numPVs+3)) - (numRecoPV+1)*(numRecoPV+1)/((numPVs+2)*(numPVs+2)) );

  float tdrPVeff = double(hisTDRPrimaryVertexVsTrueZ0_->GetEntries())/double(hisPrimaryVertexTrueZ0_->GetEntries());
  float tdrPVeff_err = sqrt((numTDRPV+1)*(numTDRPV+2)/((numPVs+2)*(numPVs+3)) - (numTDRPV+1)*(numTDRPV+1)/((numPVs+2)*(numPVs+2)) );

  cout << "PrimaryVertex Finding Efficiency = "<< recoPVeff << " +/- "<<recoPVeff_err << " Technical Proposal Algo "<< tdrPVeff << " +/- "<<tdrPVeff_err << endl;

  
  // cout << "================= L1TrkMET Trigger =============================" << endl;
  for (unsigned int i = 0; i < 4; ++i){
    float genmet = 25. + i*25.;
    float met_steps = (genmet-10.)/10;

    // cout << "********** GenMET > " << genmet << " GeV << ************" << endl;
    ostringstream name;
    name << "GenMET" <<genmet<<".txt";
    float sigEvents = double(noSignalEvents[i]);
    float bkgEvents = double(noBackgroundEvents[i]);

    hisMETevents_[i]->Fill(0.5, sigEvents);
    hisMETevents_[i]->Fill(21.5, bkgEvents);
    hisMETevents_[i]->GetXaxis()->SetBinLabel(1,"Signal Events");
    hisMETevents_[i]->GetXaxis()->SetBinLabel(22,"Bkg Events");

    for (unsigned int j = 0; j < 10; ++j){
      float cutmet = 10. + j*met_steps;
      // cout << "... L1TrkMet > "<<cutmet<<" GeV ..."<< endl;
      float efficiency = 0.;
      float rejection = 0.;
      float efficiency_tdr = 0.;
      float rejection_tdr =  0.;
      float sigma_eff = 0., sigma_eff_tdr = 0., sigma_rej = 0., sigma_rej_tdr = 0.; 

      float recoEvents = double(noRecoSignalEvents[i][j]);
      float tdrEvents = double(noTDRSignalEvents[i][j]);

      if(noSignalEvents[i] > 0){
        efficiency = double(noRecoSignalEvents[i][j])/double(noSignalEvents[i]);
        efficiency_tdr = double(noTDRSignalEvents[i][j])/double(noSignalEventsTDR[i]);
     
        sigma_eff = sqrt( (recoEvents+1)*(recoEvents+2)/((sigEvents+2)*(sigEvents+3)) - (recoEvents+1)*(recoEvents+1)/((sigEvents+2)*(sigEvents+2)) );
        sigma_eff_tdr = sqrt( (tdrEvents+1)*(tdrEvents+2)/((sigEvents+2)*(sigEvents+3)) - (tdrEvents+1)*(tdrEvents+1)/((sigEvents+2)*(sigEvents+2)) );
      }

      float recoBkgEvents = double(noRecoBackgroundEvents[i][j]);
      float tdrBkgEvents = double(noTDRBackgroundEvents[i][j]);

      hisMETevents_[i]->Fill(1.5+j, recoEvents);
      hisMETevents_[i]->Fill(11.5+j, tdrEvents);
      hisMETevents_[i]->Fill(22.5+j, recoBkgEvents);
      hisMETevents_[i]->Fill(32.5+j, tdrBkgEvents);

      ostringstream label ;
      label << "RecoSignals"<<cutmet;
      hisMETevents_[i]->GetXaxis()->SetBinLabel(j+2,label.str().c_str());
      label.clear();
      label.str("");
      label << "TPSignals"<<cutmet;
      hisMETevents_[i]->GetXaxis()->SetBinLabel(j+12,label.str().c_str());
      label.clear();
      label.str("");
      label << "RecoBkg"<<cutmet;
      hisMETevents_[i]->GetXaxis()->SetBinLabel(j+23,label.str().c_str());
      label.clear();
      label.str("");
      label << "TPBkg"<<cutmet;
      hisMETevents_[i]->GetXaxis()->SetBinLabel(j+33,label.str().c_str());


      if(noBackgroundEvents[i] > 0){
        rejection = double(noRecoBackgroundEvents[i][j])/double(noBackgroundEvents[i]);
        rejection_tdr = double(noTDRBackgroundEvents[i][j])/double(noBackgroundEventsTDR[i]);

        sigma_rej = sqrt( (recoBkgEvents+1)*(recoBkgEvents+2)/((bkgEvents+2)*(bkgEvents+3)) - (recoBkgEvents+1)*(recoBkgEvents+1)/((bkgEvents+2)*(bkgEvents+2)) );
        sigma_rej_tdr = sqrt( (tdrBkgEvents+1)*(tdrBkgEvents+2)/((bkgEvents+2)*(bkgEvents+3)) - (tdrBkgEvents+1)*(tdrBkgEvents+1)/((bkgEvents+2)*(bkgEvents+2)) );
               
      }

      

      // cout << "Signal Efficiency : "<< efficiency << " +/- "<< sigma_eff << " ("<< noRecoSignalEvents[i][j]<<"/"<<noSignalEvents[i]<<") Bkg Rejection Efficiency : "<< rejection << " +/- " << sigma_rej << "("<< noRecoBackgroundEvents[i][j]<<"/"<<noBackgroundEvents[i]<< ")"<<endl;
      // cout << "TP Algo: Signal Efficiency : "<< efficiency_tdr << " +/- "<< sigma_eff_tdr <<" ("<< noTDRSignalEvents[i][j]<<"/"<<noSignalEvents[i]<<") Bkg Rejection Efficiency : "<< rejection_tdr <<" +/- " << sigma_rej_tdr << " ("<< noTDRBackgroundEvents[i][j]<<"/"<<noBackgroundEvents[i]<< ")"<<endl;

      grMET_[i]->SetPoint(j, efficiency, rejection);
      grMET_[i]->SetPointError(j, sigma_eff, sigma_rej);
    
      grMET_tdr_[i]->SetPoint(j, efficiency_tdr, rejection_tdr);
      grMET_tdr_[i]->SetPointError(j, sigma_eff_tdr, sigma_rej_tdr);

    }
  }
}

} // end ns l1tVertexFinder
