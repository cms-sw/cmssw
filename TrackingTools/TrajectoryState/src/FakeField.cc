#include "TrackingTools/TrajectoryState/interface/FakeField.h"

// leaks memory... only for testing
MagneticField* TrackingTools::FakeField::Field::theField = new ConcreteField;
