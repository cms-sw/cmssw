#ifndef __TMTrackTrigger_VertexFinder_Histos_h__
#define __TMTrackTrigger_VertexFinder_Histos_h__


#include <map>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"



class TH1F;
class TH2F;
class TProfile;
class TGraphAsymmErrors;
class TGraphErrors;
class TEfficiency;


namespace vertexFinder {

class InputData;
class L1fittedTrack;
class Settings;
class VertexFinder;

class Histos {

public:
  // Store cfg parameters.
  Histos(const Settings* settings) : settings_(settings) {}

  ~Histos(){}

  // Book all histograms
  void book();

  /// Fill histograms relating to vertex reconstruction performance.
  void fillVertexReconstruction(const InputData& inputData, const VertexFinder& vf);
  
  void endJobAnalysis();

private:

  // Book histograms for specific topics.
  void bookVertexReconstruction();

  void makeEfficiencyPlot( TFileDirectory &inputDir, TGraphAsymmErrors* outputEfficiency, TH1F* pass, TH1F* all, TString name, TString title );


 private:

  const Settings *settings_; // Configuration parameters.

  edm::Service<TFileService> fs_;
  // Histograms for Vertex Reconstruction
  
  TH1F* hisNoRecoVertices_;
  TH1F* hisNoPileUpVertices_;
  TH1F* hisRecoVertexZ0Resolution_;
  TH1F* hisRecoVertexPTResolution_;
  TH2F* hisNoRecoVsNoTruePileUpVertices_;  
  TH2F* hisRecoVertexMETVsTrueMET_;
  TH2F* hisNoTracksFromPrimaryVertex_;
  TProfile* hisRecoVertexPTResolutionVsTruePt_;
  TH2F* hisNoTrueTracksFromPrimaryVertex_;
  TH1F* hisRecoPrimaryVertexZ0width_;
  TH1F* hisRecoPileUpVertexZ0width_;
  TH1F* hisRecoVertexZ0Spacing_;
  TH1F* hisPrimaryVertexZ0width_;  
  TH1F* hisPileUpVertexZ0_;
  TH1F* hisPileUpVertexZ0width_;
  TH1F* hisPileUpVertexZ0Spacing_;
  TH1F* hisRecoPileUpVertexZ0resolution_;
  TH1F* hisRatioMatchedTracksInPV_;
  TH1F* hisFakeTracksRateInPV_;
  TH1F* hisTrueTracksRateInPV_;
  TH2F* hisRecoVertexPTVsTruePt_;
  TH1F* hisUnmatchZ0distance_;
  TH1F* hisUnmatchZ0MinDistance_;
  TH1F* hisUnmatchPt_      ;
  TH1F* hisUnmatchEta_     ;
  TH1F* hisUnmatchTruePt_  ;
  TH1F* hisUnmatchTrueEta_ ;
  TH1F* hisLostPVtracks_      ;
  TH1F* hisUnmatchedPVtracks_ ;
  TH1F* hisNumVxIterations_;
  TH1F* hisNumVxIterationsPerTrack_;
  TH1F* hisCorrelatorInputTracks_;
  TH1F* hisCorrelatorTPInputTracks_;
  TH1F* hisCorrelatorInputVertices_;
  TH1F* hisCorrelatorTPInputVertices_;
  TProfile* hisTrkMETvsGenMET_;
  TProfile* hisRecoTrkMETvsGenMET_;
  TProfile* hisTDRTrkMETvsGenMET_;

  TH1F* hisRecoPrimaryVertexVsTrueZ0_;
  TH1F* hisTDRPrimaryVertexVsTrueZ0_;
  TH1F* hisPrimaryVertexTrueZ0_;
  TH1F* hisRecoVertexMET_;
  TH1F* hisRecoVertexPT_;
  TH1F* hisRecoPileUpVertexPT_;
  TH1F* hisRecoVertexOffPT_;
  TH1F* hisRecoVertexTrackRank_;

  TProfile* hisRecoPrimaryVertexResolutionVsTrueZ0_;
  TProfile* hisTDRPrimaryVertexResolutionVsTrueZ0_;

  TH1F* hisTDRVertexZ0Resolution_           ;
  TH1F* hisTDRVertexPTResolution_           ;
  TProfile* hisTDRVertexPTResolutionVsTruePt_   ;
  TH2F* hisTDRVertexPTVsTruePt_             ;
  TH2F* hisTDRVertexMETVsTrueMET_           ;
  TH2F* hisTDRNoTracksFromPrimaryVertex_    ;
  TH2F* hisTDRNoTrueTracksFromPrimaryVertex_;
  TH1F* hisTDRPrimaryVertexZ0width_         ;
  TH1F* hisRatioMatchedTracksInTDRPV_       ;
  TH1F* hisFakeTracksRateInTDRPV_           ;
  TH1F* hisTrueTracksRateInTDRPV_           ;
  TH1F* hisTDRUnmatchZ0distance_            ;
  TH1F* hisTDRUnmatchZ0MinDistance_         ;
  TH1F* hisTDRUnmatchPt_                    ;
  TH1F* hisTDRUnmatchEta_                   ;
  TH1F* hisTDRUnmatchTruePt_                ;
  TH1F* hisTDRUnmatchTrueEta_               ;
  TH1F* hisTDRUnmatchedPVtracks_            ;
  TH1F* hisUnmatchedVertexZ0distance_       ;
  TH1F* hisTDRUnmatchedVertexZ0distance_    ;
  TH1F* hisTDRVertexMET_;
  TH1F* hisTDRVertexPT_;
  TH1F* hisTDRPileUpVertexPT_;
  TH1F* hisTDRVertexOffPT_;
  
  TH1F* hisGenVertexPt_;
  TH1F* hisGenTkVertexPt_;

  TH1F* hisGenVertexMET_;
  TH1F* hisGenTkVertexMET_;

  TH1F* hisGenVertexTrackPt_;
  TH1F* hisGenVertexNumTracks_;

  TH1F* hisRecoVertexVsNumGenTracks_;
  TH1F* hisRecoVertexVsGenVertexPt_;
  TH1F* hisRecoGenuineVertexVsGenTkVertexPt_;

  TH1F* hisRecoVertexVsGenTkVertexPtForEff_;
  TH1F* hisRecoVertexVsGenTkVertexMETForEff_;

  TH1F* hisRecoVertexVsGenMET_;
  TH1F* hisRecoGenuineVertexVsGenMET_;
  TH1F* hisRecoGenuineVertexVsGenTkMET_;

  TH1F* hisPUVertexPt_;
  TH1F* hisPUTkVertexPt_;
  TH1F* hisPUVertexTrackPt_;
  TH1F* hisPUVertexNumTracks_;

  TProfile* hisRecoVertexMETResolution_;

  std::vector<TH1F*> hisMETevents_;
  std::vector<TH1F*> hisPTevents_;


  std::vector<TGraphErrors*> grMET_;
  std::vector<TGraphErrors*> grMET_tdr_;
  std::vector<TH1F*> hisRecoVertexVsGenTkMET_;
  std::vector<TH1F*> hisRecoVertexVsGenTkVertexPt_;



  TEfficiency* PVefficiencyVsTrueZ0_;
  TEfficiency* tdrPVefficiencyVsTrueZ0_;


  std::vector<unsigned int> noSignalEvents;
  std::vector<unsigned int> noBackgroundEvents;
  std::vector<unsigned int> noSignalEventsTDR;
  std::vector<unsigned int> noBackgroundEventsTDR;
  std::vector<std::vector<unsigned int> > noRecoSignalEvents;
  std::vector<std::vector<unsigned int> > noRecoBackgroundEvents;
  std::vector<std::vector<unsigned int> > noTDRSignalEvents;
  std::vector<std::vector<unsigned int> > noTDRBackgroundEvents;
  unsigned int noEvents;
};

} // end ns vertexFinder

#endif
