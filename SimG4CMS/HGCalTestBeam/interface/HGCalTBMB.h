#ifndef SimG4CMS_HGCalTestBeam_HGCalTBMB_h
#define SimG4CMS_HGCalTestBeam_HGCalTBMB_h

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class BeginOfTrack;
class G4Step;
class EndOfTrack;
class G4VTouchable;

#include <TH1F.h>
#include <string>
#include <vector>

class HGCalTBMB : public SimWatcher, 
                  public Observer<const BeginOfTrack*>,
                  public Observer<const G4Step*>,
                  public Observer<const EndOfTrack*> {

public:

  HGCalTBMB(const edm::ParameterSet&);
  virtual ~HGCalTBMB();
  
private:

  HGCalTBMB(const HGCalTBMB&);          // stop default
  const HGCalTBMB& operator=(const HGCalTBMB&); // ...
  
  void update(const BeginOfTrack*);
  void update(const G4Step*);
  void update(const EndOfTrack*);

  bool stopAfter(const G4Step*);
  int  findVolume(const G4VTouchable* touch, bool stop) const;
  
  std::vector<std::string>      listNames_;
  std::string                   stopName_;
  double                        stopZ_;
  unsigned int                  nList_;
  std::vector<double>           radLen_, intLen_, stepLen_;
  std::vector<TH1D*>            me100_, me200_, me300_;
};

#endif
