/*
 * This class taken for the pakage MBDigitizer implements the 
 * Rovelli-Gresele parametrization used in ORCA 6. It is included here 
 * for comparison with the current parametrization.
 *  
 */

#ifndef DTBNORMPARAM_H
#define DTBNORMPARAM_H

/** \class DTBNormParam
 *
 * Provide time vs x correction functions for different normal B field values
 *
 * \author P. Ronchese           INFN - PADOVA
 *
 * Modification:
 *    01-Oct-2002 SL Doxygenate
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */

/* C++ Headers */

/* ====================================================================== */

/* Class DTBNormParam Interface */

class DTBNormParam{

  public:

    /** Constructor Field=0*/ 
    DTBNormParam();
    /** Constructor Field normal in Tesla*/ 
    DTBNormParam(float bnorm);

    /** Destructor (empty)*/ 
    ~DTBNormParam();

    /* Operations */ 
    /// return the correction for Bnorm field
    float tcor(float xpos) const;

  private:
    // tables of function parameters for 10 normal B filed values
    static float table_offsc[11];
    static float table_coeff[11];

    /// private class to hold parameters for a Bnorm bin
    class ParamFunc {

      public:

        ParamFunc();
        ParamFunc(int bin);
        ~ParamFunc();

        // reset
        //   void set(int bin);

        // function  to compute drift time correction
        float tcor(float xpos) const;

        // functions to compute normal B field component difference 
        inline
          float dist(float bnorm)           const  // to a given value
          { return          bnorm-bin_bnorm; }
        inline
          float dist(const ParamFunc& func) const  // to another bin
          { return func.bin_bnorm-bin_bnorm; }

      private:

        float  bin_bnorm;
        float* offsc;
        float* coeff;

    };

    // Bnorm value and function parameters for lower/higher Bnorm bins
    float _bnorm;
    ParamFunc l_func;
    ParamFunc h_func;

    friend class ParamFunc;


  protected:

};
#endif // MUBARBNORMPARAM_H
