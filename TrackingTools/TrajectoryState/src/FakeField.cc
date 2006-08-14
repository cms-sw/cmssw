#include "TrackingTools/TrajectoryState/interface/FakeField.h"

// FIXME VERTEX leaks memory... only for testing
MagneticField* TrackingTools::FakeField::Field::theField = new ConcreteField;
