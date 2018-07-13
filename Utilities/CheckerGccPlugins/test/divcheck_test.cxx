// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

float a1;
float a2;
float a4;
float a6;
float a7;
float a8;
double get();

void foo(float x, float y)
{
  const double den = 123;
  const double den2 = 456;
  a1 = x / y;
  a2 = x / 3.2;
  a4 += x / den;
  a4 += x * (1./den2);
  float a3 = x * (1./ 3.);
  float a5 = a2;

  a6 = 1.0 / a5;
  a7 = 2.0 / a5;

  for (int i=0; i < 10; i++) {
    float foo = a3 / (i+1);
    a4 += foo / a3;
    a4 += 1.0 / a1;
  }

  a4 += a3;
  a8 = 3.0 / a5;
  a4 += a8  / 4.;
}



float bar1(float x)
{
  float y __attribute__((unused)) = x/10;
  return y;
}


float bar2(float x)
{
  float y1 = 10/x;
  float y2 __attribute__ ((unused)) = 20/x;
  return y1+y2;
}


float bar3(float x)
{
  float y1 __attribute__ ((unused)) = 10/x;
  float y2 = 20/x;
  return y1+y2;
}


float bar4(float x)
{
  float x1 = x;
  float y = 0;
  for (int i=0; i < 10; i++) {
    float y1 __attribute__ ((unused)) = float(i)/x1;
    y += y1;
  }
  return y;
}


float bar5(float x)
{
  float y = 0;
  for (int i=0; i < 10; i++) {
    float y1 = float(i)/x;
    y += y1;
  }
  return y;
}

float bar6 ()
{
  double C = get();
  double B = get();
  double A = get();
  const double rmin2 = C - B*B/A;
  const double b_a = B/A;
  return rmin2 + b_a;
}


void bar7f(double);
static const double bar7a = 10;
static const double bar7b = 20;
void bar7 ()
{
  bar7f(2.5*bar7a/bar7b);
}


constexpr double bar8() { return 10; }
double bar9() { return bar8() / 20.; }


// Test for a crash we had
float SetDriftSpace() {
  
  float prt;

  if(a2 != 0) prt = - a1/10.;
  else prt = + a1/10.;

  return prt;
}


// Need to propagate constant-ness of args to constexpr fns.
constexpr float bar10(float a, float b) { return a+b; }
void bar11() {
  a1 = a2 / bar10(a4, 3.0);  // shouldn't get a warning here
  a6 = a2 / bar10 (1, 4);    // should get a warning here
}
