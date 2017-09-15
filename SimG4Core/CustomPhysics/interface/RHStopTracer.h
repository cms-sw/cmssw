#ifndef SimG4Core_RHStopTracer_H
#define SimG4Core_RHStopTracer_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include <regex>

class BeginOfRun;
class BeginOfEvent;
class BeginOfTrack;
class EndOfTrack;
class G4Step;
class G4ParticleDefinition;

class RHStopTracer :  public SimProducer,
		      public Observer<const BeginOfRun *>, 
		      public Observer<const BeginOfEvent *>, 
		      public Observer<const BeginOfTrack *>,
		      public Observer<const EndOfTrack *>
{
 public:
  RHStopTracer(edm::ParameterSet const & p);
  ~RHStopTracer() override;
  void update(const BeginOfRun *) override;
  void update(const BeginOfEvent *) override;
  void update(const BeginOfTrack *) override;
  void update(const EndOfTrack *) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
 private:
  struct StopPoint {
    StopPoint (const std::string& fName, double fX, double fY, double fZ, double fT, int fId, double fMass, double fCharge) 
    : name(fName), x(fX), y(fY), z(fZ), t(fT), id(fId), mass(fMass), charge(fCharge) 
    {}
    std::string name;
    double x;
    double y;
    double z;
    double t;
    int id;
    double mass;
    double charge;
  };
  bool mStopRegular;
  double mTraceEnergy;
  int minPdgId;
  int maxPdgId;
  int otherPdgId;
  std::string mTraceParticleName;
  std::regex rePartName;
  std::vector <StopPoint> mStopPoints;
};

#endif
