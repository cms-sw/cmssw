#ifndef extraPythia_h
#define extraPythia_h

#define PYEXEC pyexec_
extern "C" {
  void PYEXEC();
}

#define PYGIVE pygive_
extern "C" {
  void PYGIVE(const char*,int length);
}

#define pyexec pyexec_
#define py1ent py1ent_

extern "C" {
  void pyexec();
  void py1ent(int&,int&,double&,double&,double&);
}

void call_pyexec();
void call_py1ent(int,int,double,double,double);

#endif
