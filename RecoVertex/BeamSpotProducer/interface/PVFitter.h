#ifndef PVFitter_H
#define PVFitter_H

/**_________________________________________________________________
   class:   PVFitter.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)
 
 version $Id: PVFitter.h,v 1.6 2010/06/04 00:05:36 jengbou Exp $

 ________________________________________________________________**/

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BSFitter.h"

#include "RecoVertex/BeamSpotProducer/interface/BeamSpotFitPVData.h"

// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"

#include <fstream>

namespace reco {
  class Vertex;
}

class PVFitter {
 public:
  PVFitter() {}
  PVFitter(const edm::ParameterSet& iConfig);
  virtual ~PVFitter();

  void readEvent(const edm::Event& iEvent);
  
  double getWidthX() { return fwidthX; }
  double getWidthY() { return fwidthY; }
  double getWidthZ() { return fwidthZ; }
  double getWidthXerr() { return fwidthXerr; }
  double getWidthYerr() { return fwidthYerr; }
  double getWidthZerr() { return fwidthZerr; }
  
  void FitPerBunchCrossing() { fFitPerBunchCrossing = true; }
  bool runBXFitter();
  bool runFitter(); 
  void resetLSRange() { fbeginLumiOfFit=fendLumiOfFit=-1; }
  void resetRefTime() { freftime[0] = freftime[1] = 0; }
  void dumpTxtFile();
  void resetAll() {
    resetLSRange();
    resetRefTime();
    pvStore_.clear();
    bxMap_.clear();
    dynamicQualityCut_ = 1.e30;
    hPVx->Reset();
    hPVy->Reset();
    fbeamspot = reco::BeamSpot();
    fbspotMap.clear();
  };
  reco::BeamSpot getBeamSpot() { return fbeamspot; }
  std::map<int, reco::BeamSpot> getBeamSpotMap() { return fbspotMap; }
  bool IsFitPerBunchCrossing() { return fFitPerBunchCrossing; }
  int* getFitLSRange() {
    int *tmp=new int[2];
    tmp[0] = fbeginLumiOfFit;
    tmp[1] = fendLumiOfFit;
    return tmp;
  }
  /// reduce size of primary vertex cache by increasing quality limit
  void compressStore ();
  /// vertex quality measure
  double pvQuality (const reco::Vertex& pv) const;
  /// vertex quality measure
  double pvQuality (const BeamSpotFitPVData& pv) const;
  int getNPVs() { return pvStore_.size(); }
  void getNPVsperBX() {
    for ( std::map<int,std::vector<BeamSpotFitPVData> >::const_iterator pvStore = bxMap_.begin(); 
	  pvStore!=bxMap_.end(); ++pvStore) {

      std::cout << "bx " << pvStore->first << " NPVs = " << (pvStore->second).size() << std::endl;
    }
  }

 private:

  reco::BeamSpot fbeamspot;
  std::map<int,reco::BeamSpot> fbspotMap;
  bool fFitPerBunchCrossing;

  std::ofstream fasciiFile;

  bool debug_;
  bool do3DFit_;
  edm::InputTag vertexLabel_;
  bool writeTxt_;
  std::string outputTxt_;

  unsigned int maxNrVertices_;
  unsigned int minNrVertices_;
  double minVtxNdf_;
  double maxVtxNormChi2_;
  unsigned int minVtxTracks_;
  double minVtxWgt_;
  double maxVtxR_;
  double maxVtxZ_;
  double errorScale_;
  double sigmaCut_;         
  
  int frun;
  int flumi;
  std::time_t freftime[2];

  TH2F* hPVx; TH2F* hPVy; 

  //bool saveNtuple_;
  //bool saveBeamFit_;
  //std::string outputfilename_;
  //TFile* file_;
  //TTree* ftree_;

  //beam fit results
  //TTree* ftreeFit_;
  int frunFit;
  int fbeginLumiOfFit;
  int fendLumiOfFit;
  char fbeginTimeOfFit[32];
  char fendTimeOfFit[32];
  double fwidthX;
  double fwidthY;
  double fwidthZ;
  double fwidthXerr;
  double fwidthYerr;
  double fwidthZerr;
  
  double fx;
  double fy;
  double fz;
  double fsigmaZ;
  double fdxdz;
  double fdydz;
  double fxErr;
  double fyErr;
  double fzErr;
  double fsigmaZErr;
  double fdxdzErr;
  double fdydzErr;

  std::vector<BeamSpotFitPVData> pvStore_; //< cache for PV data
  std::map< int, std::vector<BeamSpotFitPVData> > bxMap_; // store PV data as a function of bunch crossings
  double dynamicQualityCut_;               //< quality cut for vertices (dynamic adjustment)
  std::vector<double> pvQualities_;        //< working space for PV store adjustement
};

#endif
