/* Copyright IBM Corporation, 2020
 * author : Bob Walkup
 */

/*=============================================================*/
/* Algorithm for exp(x):                                       */
/*                                                             */
/*       x*recln2 = p + r                                      */
/*       exp(x) = (2**p) * exp(r*ln2)                          */
/*                                                             */
/*       where p = integer + n-bit fraction                    */
/*       and r is a small remainder.                           */
/*                                                             */
/*       The integer part of p goes in the exponent field,     */
/*       the n-bit fraction is handled with a lookup table.    */
/*       The remainder term gets a Taylor expansion.           */
/*                                                             */
/* Use an 8-bit fraction, and a 5th order Taylor series.       */
/*                                                             */
/* p = (2**44 + 2**43) + (x*recln2) - (2**44 + 2**43)          */
/* gives 8-bit accuracy in the n-bit fraction                  */
/*=============================================================*/
#include <math.h>
#include "lutexp.h"

double lutexp(double x)
{
   int hind;
   double result, hfactor;
   double recln2 = 1.44269504088896340736;
   double twop44_plus_twop43 = 2.6388279066624000e+13;

   struct uPair { unsigned lo ; unsigned hi ; };

   union { double d; struct uPair up; } X;
   union { double d; struct uPair up; } Result;

   double t, poly;
   double f0, f1, f2, f3, f4;
   double c2 = 1.0/2.0;
   double c3 = 1.0/6.0;
   double c4 = 1.0/24.0;
   double c5 = 1.0/120.0;

   /*--------------------------------------------------------*/
   /* multiply the input value by the reciprocal of ln(2)    */
   /* and shift by 2**44 + 2**43; save the exponent          */
   /*--------------------------------------------------------*/
   X.d = x*recln2 + twop44_plus_twop43;
   Result.up.hi = ( ( (X.up.lo >> 8) + 1023 )  << 20 ) & 0x7ff00000;
   Result.up.lo = 0;

   /*--------------------------------------------------*/
   /* compute the small remainder for the polynomial   */
   /* use the last 8 bits of the shifted X as an index */
   /*--------------------------------------------------*/
   t = x - (X.d - twop44_plus_twop43)*M_LN2;
   hind = X.up.lo & 0x000000ff;
   hfactor = Result.d * exp_table[hind];

// /*---------------------------------------*/
// /* use a polynomial expansion for exp(t) */
// /*---------------------------------------*/
// poly = 1.0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*c5))));

   /*----------------------------------------------------*/
   /* for a single call, better to factor the polynomial */
   /*----------------------------------------------------*/
   f0 = 1.0 + t;
   f2 = t*t;
   f4 = f2*f2;
   f1 = c2 + t*c3;
   f3 = c4 + t*c5;

   poly = (f0 + f1*f2) + f3*f4;

   /*---------------------------------*/
   /* construct the result and return */
   /*---------------------------------*/
   result = hfactor* poly;

   /*-----------------------------------*/
   /* check input value for valid range */
   /*-----------------------------------*/
   if      (x < -709.0) return 0.0;
   else if (x >  709.0) return HUGE_VAL;
   else return result;
}
