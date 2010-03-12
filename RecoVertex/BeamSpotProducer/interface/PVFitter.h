#ifndef PVFitter_H
#define PVFitter_H

/**_________________________________________________________________
   class:   PVFitter.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
         Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)
 
 version $Id: PVFitter.h,v 1.17 2010/02/20 02:49:00 jengbou Exp $

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
  
  bool runFitter(); 
  void resetLSRange() { fbeginLumiOfFit=fendLumiOfFit=-1; }
  void dumpTxtFile();
  reco::BeamSpot getBeamSpot() { return fbeamspot; }
  int* getFitLSRange() {
    int *tmp=new int[2];
    tmp[0] = fbeginLumiOfFit;
    tmp[1] = fendLumiOfFit;
    return tmp;
  }
  
 private:

  reco::BeamSpot fbeamspot;
  std::ofstream fasciiFile;

  bool debug_;
  bool do3DFit_;
  edm::InputTag vertexLabel_;
  bool writeTxt_;
  std::string outputTxt_;

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
    
  TH2F* hPVx; TH2F* hPVy; 

  bool saveNtuple_;
  bool saveBeamFit_;
  std::string outputfilename_;
  TFile* file_;
  TTree* ftree_;

      //beam fit results
  TTree* ftreeFit_;
  int frunFit;
  int fbeginLumiOfFit;
  int fendLumiOfFit;
  char fbeginTimeOfFit[30];
  char fendTimeOfFit[30];
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
};

#endif
