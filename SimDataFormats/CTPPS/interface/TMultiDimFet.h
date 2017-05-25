#ifndef SimRomanPot_CTPPSOpticsParameterisation_TMultiDimFet_h
#define SimRomanPot_CTPPSOpticsParameterisation_TMultiDimFet_h

#include "TMultiDimFit.h"
#include <map>

class TMultiDimFet : public TMultiDimFit
{
  public:
    TMultiDimFet() : TMultiDimFit() {}
    TMultiDimFet( const TMultiDimFit& mdf ) : TMultiDimFit( mdf ) {}
    TMultiDimFet( int dimension, TMultiDimFit::EMDFPolyType type=kMonomials, Option_t* option="" ) : TMultiDimFit( dimension, type, option ) {}

    void FindParameterization( double precision );
    void ReleaseMemory();

    void ReducePolynomial( double error );
    void ZeroDoubiousCoefficients( double error );

    //void PrintPolynomialsSpecial( Option_t *option="m" ) const;

  private:
    int fMaxFunctionsTimesNVariables;

  ClassDef( TMultiDimFet, 1 );
};

#endif
