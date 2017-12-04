#include "FWCore/Utilities/interface/thread_safety_macros.h"

// is ok, because const-qualified
const static int g_staticConst = 23;
static int const& g_ref_staticConst = g_staticConst;
static int const* g_ptr_staticConst = &g_staticConst;


// results in a warning by GlobalStaticChecker
CMS_THREAD_SAFE static int g_static;
static int * g_ptr_static = &g_static;

class ClassTest {
   public:
      explicit ClassTest();
      ~ClassTest();

void testConst() const;

   private:

      mutable int m_testMutable;
      int * m_testPointer;
      int m_testInteger;

};

void
ClassTest::testConst() const
{
    // 1) reported by class checker
    m_testMutable = 23;

    // 2) compiles, not reported
    (*m_testPointer) = 23;

    // 3) compiles, not reported
    int * localPtr = m_testPointer;
    (*localPtr) = 23;

    // 4) will not compile
    // error: invalid conversion from 'const int*' to 'int*'
    //int * localPtrToInt = &m_testInteger;
} 

class Thing
{
private:

int num;

public:

Thing(): num{0} {}
int getnum() {return num;}
void putnum(int x) {num=x;}
};

class Foo
{

private:

int Var_;
int & RVar_=Var_;
int * PVar_=&RVar_;
Thing * T_p;

public:

Foo(): Var_{0}{T_p= new Thing;}
void func1(int  x) {return;} //OK
void func2(int &x) {return;} // cound be bad 
void func3(int *x) {return;} // could be bad
void func4(int const *x) {return;} //  OK
void func5(int * const x) {return;} // could be bad
void func6(int const &x) {return;} //OK
void nonConstFunc() { Var_ = 5;}
void constFunc() const { return;}
int * nonConstAccess() const {return PVar_;} //bad
int & nonConstRefAccess() const { return  RVar_; } //bad ?
int const * constAccess() const {return PVar_;} //OK
int const & constRefAccess() const { return RVar_; } //OK ?
Thing * getThing() { return T_p; }

};

class Bar
{
CMS_THREAD_SAFE static int si_;
static void const modifyStatic(int &x) {si_=x;}
private:
Bar(): ci_{0},ipc_{&i_},icp_{&i_},ir_{i_},icr_{ci_} {}
const int ci_;
int i_;
int const * icp_;
int * ip_;
int * const ipc_;
int & ir_;
int const & icr_;
Foo foo_;
public:
void modifyMember()  { i_ = 5;}
void indirectModifyMember()  { modifyMember();}
void recursiveCaller(int i) {if (i == 0) return; recursiveCaller(--i);}

void method1(int &x) {return;}
void method2(const int &x) const {return;}

void produce() 
	{
	Foo * foo = new Foo;
	int I=0;
	int * PI;
	int const * CPI;
	foo->func1(i_);
	foo->func1(ci_);
	foo->func1(ir_);
	foo->func1(icr_);
	foo->func1(I);
	foo->func2(i_);
	foo->func2(ir_);
	foo->func2(I);
	foo->func6(i_);
	foo->func6(ir_);
	foo->func6(I);
	foo->nonConstFunc();
	foo_.nonConstFunc(); //should fail member data (object) call non const functions 
	foo_.constFunc(); //OK because const won't modify self
	foo->getThing()->getnum();
	foo_.getThing()->getnum();
	method1(i_);
	method1(I);
	modifyStatic(I);
	modifyMember();
	indirectModifyMember();
	recursiveCaller(1);
	PI=foo_.nonConstAccess(); //should fail returns pointer to member data that is non const qualified
	CPI=foo_.constAccess(); // OK because returns pointer to member data that is const qualified
	if (*PI==*CPI) I++;
	}

void method3() const
	{
	Foo foo;
	int I=0;
	Bar bar;
	foo.func1(i_);
	foo.func1(ci_);
	foo.func1(ir_);
	foo.func1(icr_);
	foo.func1(I);
	foo.func2(ir_);
	foo.func2(I);
	foo.func6(i_);
	foo.func6(ir_);
	foo.func6(I);
	foo_.nonConstRefAccess();
	foo_.constRefAccess();
//	foo_.nonConstFunc();
	foo_.nonConstAccess();
	foo_.constAccess();
	if (i_) method2(i_);
	bar.produce();
// will produce a warning only by ConstCastAwayChecker
	int & ir = (int &) (icr_);
	int & cir = const_cast<int &>(icr_);
	int * ip = (int *) (icp_);
	int * cip = const_cast<int *>(icp_);
// must not produce a warning
	int const& ira = (int const&)(icr_);
// will produce a warning by StaticLocalChecker
        CMS_THREAD_SAFE static int evilStaticLocal = 0;
	static int & intRef = evilStaticLocal;
	static int * intPtr = & evilStaticLocal;
// no warnings here
	static const int c_evilStaticLocal = 0;
	static int const& c_intRef = evilStaticLocal;
	static int const* c_intPtr = &evilStaticLocal;
	static const int * c_intPtr_equivalent = &evilStaticLocal;
	static int const* const* c_intPtrPtr = &( c_intPtr);
	g_static=23;
	si_=23;
	modifyStatic(I);
	}


};

int main()
{
return 0;
}



