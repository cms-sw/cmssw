#ifndef TrackingTools_FakeField_h
#define TrackingTools_FakeField_h
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/Math/interface/Point3D.h"

namespace  TrackingTools{
  namespace FakeField {
    class Field{
      public:
//       Field( MagneticField* aField): theField(aField) {};
      static  GlobalVector inTesla( const GlobalPoint& pos)
      	 {return theField->inTesla(pos);}
      static  GlobalVector inTesla( const math::XYZPoint& pos)
       	 {return theField->inTesla(GlobalPoint(pos.x(),pos.y(),pos.z()));}
      static  MagneticField* field() {return theField;}
     private:
      static MagneticField* theField;
    };
    MagneticField* Field::theField = 0;

  }
}
#endif
