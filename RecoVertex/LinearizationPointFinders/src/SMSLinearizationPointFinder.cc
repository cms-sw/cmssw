#include "RecoVertex/LinearizationPointFinders/interface/SMSLinearizationPointFinder.h"
#include "RecoVertex/VertexTools/interface/SmsModeFinder3d.h"

SMSLinearizationPointFinder::SMSLinearizationPointFinder(
    const signed int n_pairs, const SMS & sms ) :
  CrossingPtBasedLinearizationPointFinder ( 
      SmsModeFinder3d( sms ), n_pairs )
{ }

SMSLinearizationPointFinder::SMSLinearizationPointFinder(
    const RecTracksDistanceMatrix * m, const signed int n_pairs, const SMS & sms ) :
  CrossingPtBasedLinearizationPointFinder ( m , 
      SmsModeFinder3d( sms ), n_pairs )
{ }

SMSLinearizationPointFinder * SMSLinearizationPointFinder::clone()
  const
{
  return new SMSLinearizationPointFinder ( * this );
}
