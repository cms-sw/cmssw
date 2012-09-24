class Foo
{

private:

int Var_;

public:

Foo(): Var_{0}{}
void Bar1(int  x) {return;} //OK
void Bar2(int &x) {return;} // cound be bad 
void Bar3(int *x) {return;} // could be bad
void Bar4(int const *x) {return;} //  OK
void Bar5(int * const x) {return;} // could be bad
void Bar6(int const &x) {return;} //OK
void modifyMember() { Var_ = 5;}
void Bar7() { modifyMember();} //bad
void constCase() const { return;}

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

void modifyMember() { i_ = 5;}
void indirectModifyMember() { modifyMember();}

void method1(int &x) {return;}

void produce() 
	{
	Foo foo;
	int I=0;
	foo.Bar1(i_);
	foo.Bar1(ci_);
	foo.Bar1(ir_);
	foo.Bar1(icr_);
	foo.Bar1(I);
	foo.Bar2(i_);
	foo.Bar2(ir_);
	foo.Bar2(I);
	foo.Bar6(i_);
	foo.Bar6(ir_);
	foo.Bar6(I);
	foo.Bar7();
	method1(i_);
	method1(I);
	modifyMember();
	indirectModifyMember();
	}

void method3() const
	{
	Foo foo;
	int I=0;
	foo.Bar1(i_);
	foo.Bar1(ci_);
	foo.Bar1(ir_);
	foo.Bar1(icr_);
	foo.Bar1(I);
	foo.Bar2(ir_);
	foo.Bar2(I);
	foo.Bar6(i_);
	foo.Bar6(ir_);
	foo.Bar6(I);
	}


};

int main()
{
return 0;
}



