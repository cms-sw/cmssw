class Foo
{

private:

int Var_;
int * PVar_;

public:

Foo(): Var_{0}{}
void func1(int  x) {return;} //OK
void func2(int &x) {return;} // cound be bad 
void func3(int *x) {return;} // could be bad
void func4(int const *x) {return;} //  OK
void func5(int * const x) {return;} // could be bad
void func6(int const &x) {return;} //OK
void nonConstFunc() { Var_ = 5;}
void constFunc() const { return;}
int * nonConstAccess() const {return PVar_;} //bad
int const * constAccess() const {return PVar_;} //OK

};

class Bar
{
private:
Bar(): ci_{0},ipc_{&i_},ir_{i_},icr_{ci_}{}
const int ci_;
int i_;
int const * icp_;
int * ip_;
int * const ipc_;
int & ir_;
int const & icr_;
Foo foo_;
public:
void modifyMember() { i_ = 5;}
void indirectModifyMember() { modifyMember();}
void recursiveCaller(int i) {if (i == 0) return; recursiveCaller(--i);}

void method1(int &x) {return;}

void produce() 
	{
	Foo * foo;
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
	method1(i_);
	method1(I);
	modifyMember();
	indirectModifyMember();
	recursiveCaller(1);
	PI=foo_.nonConstAccess(); //should fail returns pointer to member data that is non const qualifies
	CPI=foo_.constAccess(); // OK because returns pointer to member data that is const qualified
	}

void method3() const
	{
	Foo foo;
	int I=0;
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
	}


};

int main()
{
return 0;
}



