#ifndef DAFValidator_h
#define DAFValidator_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include <vector>
#include <map>

#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TH1F.h>

class TrackerHitAssociator;
class TrackerGeometry;
class PSimHit;
 

class DAFValidator : public edm::EDAnalyzer 
{
	public:
	DAFValidator(const edm::ParameterSet& conf);
	virtual ~DAFValidator();
	
	virtual void beginRun(edm::Run & run, const edm::EventSetup& c);
	virtual void endJob();
	virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

	void analyzeHits(const TrackingParticle* tpref, 
			 const reco::Track* rtref, 
			 TrackerHitAssociator& hitassociator,
			 const edm::Ref<std::vector<Trajectory> > traj_iterator,
			 const TrackerGeometry* geom, 
			 int event) ;
	
	void analyzeHits2(const edm::Ref<std::vector<Trajectory> > traj_iterator, reco::TrackRef track);
	
	bool check(const std::vector<PSimHit>& simhits, const TrackingParticle* tpref) const;

	bool check(const std::vector<SimHitIdpr>& simhitids, const TrackingParticle* tpref) const;
	
        void fillDAFHistos(std::vector<PSimHit>& simhit, 
			   float weight, 
			   const TrackingRecHit* hit,
			   const TrackerGeometry* geom);
	
	void fillPHistos(std::vector<PSimHit>& components);
	void fillMultiHitHistos(float a, 
				const TrackingRecHit*, 
				const TrackerGeometry*);

	int fillNotMergedHisto(const std::vector<SimHitIdpr>& simhitids,
				const std::vector<PSimHit>& simhits,
				const TrackingParticle* tpref,
				float weight,
				const TrackerGeometry*) const;

	int fillMergedHisto(const std::vector<SimHitIdpr>& simhitids1,
			     const std::vector<PSimHit>& simhits1,
			     const TrackingParticle* tpref1,
			     float weight,
			     const TrackerGeometry*) const;

	float calculatepull(const TrackingRecHit* hit, 
			    PSimHit simhit,
			    const TrackerGeometry*);

	void fillMultiHitHistosPartiallyUnmatched(const std::vector<std::pair<float,const TrackingRecHit*> >& map, const TrackerGeometry* geom);
	void fillMultiHitHistosTotallyUnmatched(const std::vector<std::pair<float,const TrackingRecHit*> >& map, const TrackerGeometry* geom);
	float getType(const TrackingRecHit* hit)  const;
	GlobalPoint getGlobalPositionRec(const TrackingRecHit* hit, const TrackerGeometry* geom) const;
	GlobalPoint getGlobalPositionSim(const PSimHit hit, const TrackerGeometry* geom) const;
	GlobalPoint getGlobalPosition(const TrackingRecHit* hit, const TrackerGeometry* geom) const;

	private:
	edm::ParameterSet theConf;
	TFile* output;
	TH1F* histo_weight;
	TH2F* weight_withassociatedsimhit_vs_type;
	TH2F* weight_withassociatedsimhit_vs_r;
	TH2F* weight_withassociatedsimhit_vs_eta;
	TH1F* weight_partially_unmatched;
	TH2F* weight_vs_type_partially_unmatched;	
	TH2F* weight_vs_r_partially_unmatched;
	TH2F* weight_vs_eta_partially_unmatched;
	TH1F* weight_totally_unmatched;
        TH2F* weight_vs_type_totally_unmatched;
        TH2F* weight_vs_r_totally_unmatched;
        TH2F* weight_vs_eta_totally_unmatched; 	
	TH1F* processtype_withassociatedsimhit_merged;
	TH1F* processtype_withassociatedsimhit;
	TH1F* Hit_Histo;
	TH1F* MergedHisto;
	TH1F* NotMergedHisto;
	TH2F* weight_vs_processtype_merged;
	TH2F* weight_vs_processtype_notmerged;
	TH2F* pull_vs_weight;
	TH2F* Merged_vs_weight;
	TH2F* NotMerged_vs_weight;
	TH2F* NotMergedPos;
	TH2F* MergedPos;
	TTree* mrhit;
	//	TTree* mrhit2;
	int mergedtype;
	int notmergedtype;
	float weight;
	uint32_t detId;
	float r;
	float zeta;
	float phi;
	float hittype;
	int nevent;
	float hitlocalx;
	float hitlocaly;
	float hitlocalsigmax;
	float hitlocalsigmay;
	float hitlocalcov;
	float tsoslocalx;
	float tsoslocaly;
	float tsoslocalsigmax;
	float tsoslocalsigmay;
	float tsoslocalcov;
	float SimTracknum;
	float RecoTracknum;


	
};
#endif
