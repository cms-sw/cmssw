#ifndef Validation_RecoTrack_MTVHistoProducerAlgo_h
#define Validation_RecoTrack_MTVHistoProducerAlgo_h

/* \author B.Mangano, UCSD
 *
 * Base class which defines the interface of a generic HistoProducerAlogs
 * to be used within the MultiTrackValidator module.
 * The concrete algorithms will be plugged into the MTV to produce all
 * the validation plots that the user wants.
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <TH1F.h>
#include <TH2F.h>

class MTVHistoProducerAlgo{
 public:

 MTVHistoProducerAlgo(const edm::ParameterSet& pset, edm::ConsumesCollector && iC) : MTVHistoProducerAlgo(pset, iC){};
 MTVHistoProducerAlgo(const edm::ParameterSet& pset, edm::ConsumesCollector & iC) : pset_(pset){};
  virtual ~MTVHistoProducerAlgo() {}

  virtual void bookSimHistos(DQMStore::IBooker& ibook)=0;
  virtual void bookSimTrackHistos(DQMStore::IBooker& ibook)=0;
  virtual void bookRecoHistos(DQMStore::IBooker& ibook)=0;
  virtual void bookRecodEdxHistos(DQMStore::IBooker& ibook)=0;
  virtual void bookRecoHistosForStandaloneRunning(DQMStore::IBooker& ibook)=0;

  virtual void fill_generic_simTrack_histos(int counter,const TrackingParticle::Vector&,const TrackingParticle::Point& vertex, int bx)=0;

  virtual void fill_recoAssociated_simTrack_histos(int count,
						   const TrackingParticle& tp,
						   const TrackingParticle::Vector& momentumTP, const TrackingParticle::Point& vertexTP,
						   double dxy, double dz, int nSimHits,
                                                   int nSimLayers, int nSimPixelLayers, int nSimStripMonoAndStereoLayers,
						   const reco::Track* track,
						   int numVertices,
						   double dR)=0;

  virtual void fill_recoAssociated_simTrack_histos(int count,
						   const reco::GenParticle& tp,
						   const TrackingParticle::Vector & momentumTP, const TrackingParticle::Point & vertexTP,
						   double dxy, double dz, int nSimHits,
						   const reco::Track* track,
						   int numVertices)=0;

  virtual void fill_generic_recoTrack_histos(int count,
				     	     const reco::Track& track,
				     	     const math::XYZPoint& bsPosition,
				     	     bool isMatched,
				     	     bool isSigMatched,
				     	     bool isChargeMatched,
					     int numAssocRecoTracks,
                         	             int numVertices,
				             int nSimHits,
   					     double sharedFraction, double dR)=0;

  virtual void fill_dedx_recoTrack_histos(int count, const edm::RefToBase<reco::Track>& trackref, const std::vector< const edm::ValueMap<reco::DeDxData> *>& v_dEdx)=0;

  virtual void fill_simAssociated_recoTrack_histos(int count,
						   const reco::Track& track)=0;

  virtual void fill_trackBased_histos(int count,
		 	      	      int assTracks,
			      	      int numRecoTracks,
			      	      int numSimTracks)=0;

  virtual void fill_ResoAndPull_recoTrack_histos(int count,
						 const TrackingParticle::Vector& momentumTP,
						 const TrackingParticle::Point& vertexTP,
						 int chargeTP,
						 const reco::Track& track,
						 const math::XYZPoint& bsPosition)=0;

  virtual void finalHistoFits(int counter)=0;


  virtual void fillProfileHistosFromVectors(int counter)=0;


 protected:
  //protected functions

  virtual double getEta(double eta)=0;

  virtual double getPt(double pt)=0;

  void doProfileX(TH2 * th2, MonitorElement* me);

  void doProfileX(MonitorElement * th2m, MonitorElement* me) {
    doProfileX(th2m->getTH2F(), me);
  }

  template<typename T> void fillPlotNoFlow(MonitorElement* h, T val){
    h->Fill(std::min(std::max(val,((T) h->getTH1()->GetXaxis()->GetXmin())),((T) h->getTH1()->GetXaxis()->GetXmax())));
  }

  void fillPlotFromVector(MonitorElement* h, std::vector<int>& vec);

  void fillPlotFromVectors(MonitorElement* h,
			   std::vector<int>& numerator,
			   std::vector<int>& denominator,
			   std::string type);

  void fillPlotFromPlots(MonitorElement* h,
			 TH1* numerator,
			 TH1* denominator,
			 std::string type);

  void BinLogX(TH1*h);

 private:
  //private data members
  const edm::ParameterSet& pset_;


};

#endif
