#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"


#include <boost/regex.hpp>

class BeginOfRun;
class BeginOfEvent;
class BeginOfTrack;
class EndOfTrack;
class G4Step;

class RHStopTracer :  public SimProducer,
		      public Observer<const BeginOfRun *>, 
		      public Observer<const BeginOfEvent *>, 
		      public Observer<const BeginOfTrack *>,
		      public Observer<const EndOfTrack *>
{
 public:
  RHStopTracer(edm::ParameterSet const & p);
  virtual ~RHStopTracer();
  void update(const BeginOfRun *);
  void update(const BeginOfEvent *);
  void update(const BeginOfTrack *);
  void update(const EndOfTrack *);
  void produce(edm::Event&, const edm::EventSetup&);
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
  boost::regex mTraceParticleNameRegex;
  std::vector <StopPoint> mStopPoints;
  bool matched (const std::string& fName) const;
};
