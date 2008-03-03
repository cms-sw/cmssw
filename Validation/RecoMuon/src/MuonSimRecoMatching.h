#ifndef Validation_RecoMuon_MuonSimRecoMatching_H
#define Validation_RecoMuon_MuonSimRecoMatching_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include <algorithm>
#include <list>

typedef TrajectoryStateOnSurface TSOS;
typedef TrackingParticleCollection TPColl;
typedef TPColl::const_iterator TPCIter;
typedef reco::MuonCollection MuColl;
typedef MuColl::const_iterator MuCIter;

template<typename QTester>
class SimRecoTable
{
 public:
  typedef std::map<double, std::pair<TPCIter, MuCIter> > QTable;
  typedef std::list<std::pair<TPCIter, MuCIter> > Pairs;
  typedef std::list<TPCIter> SimPtcls;
  
  SimRecoTable(const edm::Handle<TPColl>& simPtcls,
               const edm::Handle<MuColl>& recoMuons,
               const QTester& test)
    {
      for(TPCIter iSimPtcl = simPtcls->begin();
          iSimPtcl != simPtcls->end(); iSimPtcl++) {
        int nSimToReco = 0;

        if ( ! test(*iSimPtcl) ) continue; 

        for(MuCIter iRecoMuon = recoMuons->begin();
            iRecoMuon != recoMuons->end(); iRecoMuon++) {
          if ( ! test(*iRecoMuon) ) continue;

          double quality;
          if ( ! test(*iSimPtcl, *iRecoMuon, quality) ) continue;
          qTable_[quality] = std::make_pair(iSimPtcl, iRecoMuon);
          nSimToReco++;
        }

        if ( nSimToReco == 0 ) {
          unmatchedSimPtcls_.push_back(iSimPtcl);
        }
      }
    };
  
  // getBestMatched()
  // Fills up matched pairs sorted by quality without duplication
  // So length of returnd pairs is min(nSim, nReco)
  // Note that smaller is considerd as better (like deltaR)
  void getBestMatched(Pairs& targetPairs)
  {
    const int nMaxPair = nRecoMuons_ > nSimPtcls_ ? nRecoMuons_ : nSimPtcls_;

    targetPairs.clear();

    int nPair = 0;
    for(QTable::const_iterator iQTable = qTable_.begin();
        iQTable != qTable_.end() && nPair < nMaxPair; iQTable++) {
      bool isDuplicated = false;
      for(Pairs::const_iterator iPair = targetPairs.begin();
          iPair != targetPairs.end(); iPair++) {
        if ( iPair->first == iQTable->second.first ||
             iPair->second == iQTable->second.second ) {
          isDuplicated = true;
          break;
        }
      }
      if ( ! isDuplicated ) {
        targetPairs.push_back(iQTable->second);
        nPair++;
      }
    }
    std::cout << "# of best matching pairs = " << nPair << std::endl;
  }; 

  void getUnmatched(SimPtcls& targetList)
  {
    targetList.clear();
    targetList.assign(unmatchedSimPtcls_.begin(), unmatchedSimPtcls_.end());
  };

 protected:
  int nSimPtcls_, nRecoMuons_;
  // This is somehow tricky part
  // map type is automatically sorted by "keys"
  // std::map can be problematic when quality1 = quality2
  QTable qTable_;
  SimPtcls unmatchedSimPtcls_;
};

class MuonDeltaR
{
 public:
  MuonDeltaR(const double maxDeltaR);
  bool operator()(const TrackingParticle& simPtcl) const;
  bool operator()(const reco::Muon& recoMuon) const;
  bool operator()(const TrackingParticle& simPtcl,
                  const reco::Muon& recoMuon,
                  double& result) const;
  
 protected:
  const double maxDeltaR_;
};

/*
class MuTrkChi2 
{
public:
  MuTrkChi2(const double maxChi2,
            const edm::EventSetup& eventSetup,
            const bool onlyDiagonal = false);
  
  bool operator()(TPCIter simPtcl, MuCIter recoMuon, double& result) const;
  
 protected:
  const double maxChi2_;
  edm::ESHandle<MagneticField> theMF;
  const bool onlyDiagonal_;
  
  bool paramsAtClosest(const Basic3DVector<double> vtx,
                       const Basic3DVector<double> momAtVtx,
                       const float charge,
                       reco::TrackBase::ParameterVector& trkParams) const;
};
*/

#endif
