class Foo
{

private:

int Var;

public:

Foo() {Var=0;}
void Bar1(int  x) {return;} //OK
void Bar2(int &x) {return;} // cound be bad 
void Bar3(int *x) {return;} // could be bad
void Bar4(int const *x) {return;} //  OK
void Bar5(int * const x) {return;} // could be bad
void Bar6(int const &x) {return;} //OK

};

class Bar
{
private:

const int ci;
int i;
int const * icp;
int * ip;
int * const ipc;
int & ir;
int const & icr;

void method1(int &x) {return;}

void method2()
{
Foo foo;
int I=0;
foo.Bar1(i);
foo.Bar1(ci);
foo.Bar1(ir);
foo.Bar1(icr);
foo.Bar1(I);
foo.Bar2(i);
//foo.Bar2(ci);
foo.Bar2(ir);
//foo.Bar2(icr);
foo.Bar2(I);
method1(i);
method1(I);
}

};

int main()
{
return 0;
}



