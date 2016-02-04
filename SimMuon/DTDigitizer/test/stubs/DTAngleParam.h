
/*
 * This class taken for the pakage MBDigitizer implements the 
 * Rovelli-Gresele parametrization used in ORCA 6. It is included here 
 * for comparison with the current parametrization.
 *  
 */


#ifndef DTANGLEPARAM_H
#define DTANGLEPARAM_H

/** \class DTAngleParam
 *
 *  Provide time vs x vec(B) functions for different angle values
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

/* Class DTAngleParam Interface */

class DTAngleParam{

  public:

    /** Constructor with angle 0.*/ 
    DTAngleParam();
    /** Constructor with a given angle (degree)*/ 
    DTAngleParam(float angle);

    /** Destructor */ 
    ~DTAngleParam() ;

    /* Operations */ 
    /// return the time given x and B
    float time(float bwire, float xcoor) const;

  private:
    // tables of function parameters 
    // 19 angle values up to 10 terms in a function
    static int   table_num_terms[19];
    static int   table_pow_field[190];
    static int   table_pow_xcoor[190];
    static float table_offsc[    19];
    static float table_coeff[    190];

    /// private class to hold parameters for an angle bin
    class ParamFunc {

      public:

        ParamFunc();
        ParamFunc(int bin);
        ~ParamFunc();

        // reset
        //   void set(int bin);

        // function  to compute drift time
        float time(float bwire, float xcoor) const;

        // functions to compute angle difference 
        inline
          float dist(float angle)           const  // to a given value
          { return          angle-bin_angle; }
        inline
          float dist(const ParamFunc& func) const  // to another bin
          { return func.bin_angle-bin_angle; }

      private:

        float  bin_angle;
        int*   num_terms;
        int*   pow_field;
        int*   pow_xcoor;
        float* offsc;
        float* coeff;

    };

    // angle value and function parameters for lower/higher angle bins
    float _angle;
    ParamFunc l_func;
    ParamFunc h_func;

    friend class ParamFunc;

  protected:

};
#endif // MUBARANGLEPARAM_H
