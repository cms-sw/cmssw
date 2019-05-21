class HTrackVariables;
class HResolution;
class TFile;
class SimTrack;
class TrajectoryStateOnSurface;
class FreeTrajectoryState;

#include "TString.h"
#include <string>
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>

class HTrack {
public:
  HTrack(DQMStore::IBooker &, std::string, std::string name, std::string whereIs = "");

  double pull(double rec, double sim, double sigmarec);

  double resolution(double rec, double sim);

  void computeResolution(const FreeTrajectoryState &fts, SimTrack &simTracks, HResolution *hReso);

  void computeTDRResolution(const FreeTrajectoryState &fts, SimTrack &simTracks, HResolution *hReso);

  void computePull(const FreeTrajectoryState &fts, SimTrack &simTracks, HResolution *hReso);

  void computeResolutionAndPull(TrajectoryStateOnSurface &vtx, SimTrack &simTrack);

  void computeResolutionAndPull(const FreeTrajectoryState &fts, SimTrack &simTrack);

  void Fill(TrajectoryStateOnSurface &);
  void Fill(const FreeTrajectoryState &);
  void FillDeltaR(double);

  double computeEfficiency(HTrackVariables *sim, DQMStore::IBooker &);

private:
  HTrackVariables *hVariables;

  // Global Resolution
  HResolution *hResolution;
  HResolution *hPull;

  // TDR Resolution
  HResolution *hTDRResolution;
  HResolution *hTDRPull;

  // [5-10] GeV range
  HResolution *hResolution_5_10;
  HResolution *hTDRResolution_5_10;
  HResolution *hPull_5_10;

  // [10-40] GeV range
  HResolution *hResolution_10_40;
  HResolution *hTDRResolution_10_40;
  HResolution *hPull_10_40;

  // [40-70] GeV range
  HResolution *hResolution_40_70;
  HResolution *hTDRResolution_40_70;
  HResolution *hPull_40_70;

  // [70-100] GeV range
  HResolution *hResolution_70_100;
  HResolution *hTDRResolution_70_100;
  HResolution *hPull_70_100;

  // eta range |eta|<0.8
  HResolution *hResolution_08;
  HResolution *hTDRResolution_08;
  HResolution *hPull_08;

  // eta range 0.8<|eta|<1.2
  HResolution *hResolution_08_12;
  HResolution *hTDRResolution_08_12;
  HResolution *hPull_08_12;

  // eta range 1.2<|eta|<2.1
  HResolution *hResolution_12_21;
  HResolution *hTDRResolution_12_21;
  HResolution *hPull_12_21;

  // eta range 1.2<|eta|<2.4
  HResolution *hResolution_12_24;
  HResolution *hTDRResolution_12_24;
  HResolution *hPull_12_24;

  // eta range 1.2<eta<2.1
  HResolution *hResolution_12_21_plus;
  HResolution *hTDRResolution_12_21_plus;
  HResolution *hPull_12_21_plus;

  // eta range 1.2<eta<2.4
  HResolution *hResolution_12_24_plus;
  HResolution *hTDRResolution_12_24_plus;
  HResolution *hPull_12_24_plus;

  // eta range -2.1<eta<-1.2
  HResolution *hResolution_12_21_minus;
  HResolution *hTDRResolution_12_21_minus;
  HResolution *hPull_12_21_minus;

  // eta range -2.4<eta<-1.2
  HResolution *hResolution_12_24_minus;
  HResolution *hTDRResolution_12_24_minus;
  HResolution *hPull_12_24_minus;

  TString theName;
  TString where;

  bool doSubHisto;
};
