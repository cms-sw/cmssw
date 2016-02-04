#ifndef TrackingMaterialProducer_h
#define TrackingMaterialProducer_h
#include <string>
#include <vector>
 
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "SimG4Core/Notification/interface/Observer.h"

#include "G4LogicalVolume.hh"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"

class BeginOfJob;
class BeginOfEvent;
class BeginOfTrack;
class EndOfTrack;
class G4Step;

class G4StepPoint;
class G4VTouchable;
class G4VPhysicalVolume;
class G4LogicalVolume;

class TrackingMaterialProducer : public SimProducer,
                                 public Observer<const BeginOfJob*>,
                                 public Observer<const BeginOfEvent*>,
                                 public Observer<const BeginOfTrack*>,
                                 public Observer<const G4Step*>,
                                 public Observer<const EndOfTrack*>
{
public:
  TrackingMaterialProducer(const edm::ParameterSet&);
  virtual ~TrackingMaterialProducer();
  
private:
  void update(const BeginOfJob*);
  void update(const BeginOfEvent*);
  void update(const BeginOfTrack*);
  void update(const G4Step*);
  void update(const EndOfTrack*);
  void produce(edm::Event&, const edm::EventSetup&);
 
  bool isSelected( const G4VTouchable* touch );

private:
  bool                                  m_primaryTracks;
  std::vector<std::string>              m_selectedNames; 
  std::vector<const G4LogicalVolume *>  m_selectedVolumes;
  MaterialAccountingTrack               m_track;
  std::vector<MaterialAccountingTrack>* m_tracks;  
};

#endif // TrackingMaterialProducer_h
