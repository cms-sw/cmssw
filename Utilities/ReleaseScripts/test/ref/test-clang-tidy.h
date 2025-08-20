class BaseClass1 {
public:
  BaseClass1() {}
  virtual void override_func() = 0;
  virtual ~BaseClass1() {}
};

class BaseClass2 : public BaseClass1 {
public:
  BaseClass2() {}
  void override_func() override {}
  ~BaseClass2() override {}
};
