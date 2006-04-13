#include "TrackingTools/TrajectoryState/interface/FakeField.h"

// FIXME leaks memory... only for testing
MagneticField* TrackingTools::FakeField::Field::theField = new ConcreteField;
