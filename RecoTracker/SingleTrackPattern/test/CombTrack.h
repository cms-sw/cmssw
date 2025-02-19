#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include <TROOT.h>
#include <TFile.h>
#include <TH2F.h>

class MagneticField; 
class CompareChi2{
 public:
  CompareChi2(){};
  bool operator()( const reco::Track *tk1,
		   const reco::Track *tk2){
    return tk1->chi2()<tk2->chi2();};
};
class CombTrack : public edm::EDAnalyzer {
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
   public:
      explicit CombTrack(const edm::ParameterSet&);
      ~CombTrack();

      void AnalHits(const TrackingRecHitCollection &hits, 
		    const edm::EventSetup&);
 private:
      virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      TFile* hFile;
      TH2F *hpt,*hq,*hpz,*heta,*hphi;
      TH2F *hpt_tob,*hq_tob,*hpz_tob,*heta_tob,*hphi_tob;
      TH2F *hpt_tib,*hq_tib,*hpz_tib,*heta_tib,*hphi_tib;
      TH2F *hpt_3l,*hq_3l,*hpz_3l,*heta_3l,*hphi_3l;
      TH2F *hpt_4l,*hq_4l,*hpz_4l,*heta_4l,*hphi_4l;
      edm::ParameterSet conf_;
      const TransientTrackingRecHitBuilder *RHBuilder;
      bool ltib1,ltib2,ltob1,ltob2;
      unsigned int nlay;
      bool trinevents;

      edm::ESHandle<MagneticField> magfield;

};
  
