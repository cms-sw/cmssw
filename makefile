ObjSuf        = o
SrcSuf        = cc
IncSuf        = h
ExeSuf        =
DllSuf        = so
DepSuf        = d
OutPutOpt     = -o

OBJ_SUB_DIR   = obj
INCL_SUB_DIR  = interface
SRC_SUB_DIR   = src
DEP_SUB_DIR   = dep
BIN_SUB_DIR   = bin
LIB_SUB_DIR   = lib
TEST_SUB_DIR   = test

ifndef TOTEMRPSIMREC
TOTEMRPSIMREC = ./
endif

MAIN_OBJ_PATH      = $(TOTEMRPSIMREC)/$(OBJ_SUB_DIR)
MAIN_SRC_PATH      = $(TOTEMRPSIMREC)/$(SRC_SUB_DIR)
MAIN_BIN_PATH      = $(TOTEMRPSIMREC)/$(BIN_SUB_DIR)
MAIN_LIB_PATH      = $(TOTEMRPSIMREC)/$(LIB_SUB_DIR)
MAIN_INC_PATH      = $(TOTEMRPSIMREC)/$(INCL_SUB_DIR)
MAIN_DEP_PATH      = $(TOTEMRPSIMREC)/$(DEP_SUB_DIR)
PACKAGE_PATH       = $(TOTEMRPSIMREC)

ROOTCONF      = root-config

ROOTLIBS      = `$(ROOTCONF) --libs`
ROOTLIBS      += `$(ROOTCONF) --glibs`
ROOTINCL      = -I`$(ROOTCONF) --incdir`


# AFS resources - headers
#CMSINCL       += -I/afs/cern.ch/sw/lcg/external/XercesC/2.8.0/x86_64-slc5-gcc41-opt/include
#CMSINCL       += -I/afs/cern.ch/sw/lcg/external/clhep/2.0.4.6/x86_64-slc5-gcc41-opt/include
#CMSINCL       += -I/opt/boost_1_36_0

# AFS resources - libs
#CMSLIB        += -L/afs/cern.ch/sw/lcg/external/XercesC/2.8.0/x86_64-slc5-gcc41-opt/lib -lxerces-c -lxerces-depdom
#CMSLIB        += -L/afs/cern.ch/sw/lcg/external/clhep/2.0.4.6/x86_64-slc5-gcc41-opt/lib -lCLHEP-2.0.4.6 -lCLHEP-Vector-2.0.4.6

# local resources - libs
CMSLIB        += -lxerces-c

CMSLIB	      += $(ROOTLIBS)
CMSINCL       += $(ROOTINCL)

# Linux
CXX           = g++
CXXFLAGS      = -O3  -fPIC  -pthread #-pedantic -Wall 
LD            = g++
LDFLAGS       = -O3
SOFLAGS       = -shared
RM      	  = rm

#------------------------------------------------------------------------------

BASE_PROJECT_DIRS := $(TOTEMRPSIMREC)

#------------------------------------------------------------------------------

MAIN_STUB_PATH := $(TOTEMRPSIMREC)/stub
PROJECT_STUB_HEADERS := $(TOTEMRPSIMREC)/interface/TMultiDimFet.h $(TOTEMRPSIMREC)/interface/LHCOpticsApproximator.h $(TOTEMRPSIMREC)/interface/TNtupleDcorr.h
CINT_CONF_FILE = FitCintLinkDef.h
CINT_OUT_FILE_BASE = FitCint

#------------------------------------------------------------------------------

PROJECT_SOURCES = $(foreach dir, $(BASE_PROJECT_DIRS), $(wildcard $(dir)/$(SRC_SUB_DIR)/*.$(SrcSuf)))
PROJECT_OBJS = $(join $(patsubst %$(SRC_SUB_DIR)/, %$(OBJ_SUB_DIR)/, $(dir $(PROJECT_SOURCES))), $(addsuffix .$(ObjSuf), $(basename $(notdir $(PROJECT_SOURCES))))) $(MAIN_OBJ_PATH)/$(CINT_OUT_FILE_BASE).$(ObjSuf)
PROJECT_OBJS_READY = $(foreach dir, $(BASE_PROJECT_DIRS), $(wildcard $(dir)/$(OBJ_SUB_DIR)/*.$(ObjSuf)))
PROJECT_HEADER_FILES = $(foreach dir, $(BASE_PROJECT_DIRS), $(wildcard $(dir)/$(INCL_SUB_DIR)/*.$(IncSuf)))
PROJ_INCLUDE = $(addprefix -I, $(addsuffix /$(INCL_SUB_DIR), $(BASE_PROJECT_DIRS))) -I$(TOTEMRPSIMREC)
PROJECT_MAIN = bin/FindApproximation
PROJECT_OBJ = obj/FindApproximation.o


PROJ_DEPS = $(join $(patsubst %$(SRC_SUB_DIR)/, %$(DEP_SUB_DIR)/, $(dir $(PROJECT_SOURCES))), $(addsuffix .$(DepSuf), $(basename $(notdir $(PROJECT_SOURCES))))) $(MAIN_DEP_PATH)/$(CINT_OUT_FILE_BASE).$(DepSuf)
PROJECT_DEPS_READY = $(foreach dir, $(BASE_PROJECT_DIRS), $(wildcard $(dir)/$(DEP_SUB_DIR)/*.$(DepSuf))) $(foreach dir, $(BASE_PROJECT_DIRS), $(wildcard $(dir)/$(TEST_SUB_DIR)/*.$(DepSuf)))

PROJECT_TEST_SRC = $(foreach dir, $(BASE_PROJECT_DIRS), $(wildcard $(dir)/$(TEST_SUB_DIR)/*.$(SrcSuf)))
PROJECT_TEST_OBJ = $(addsuffix .$(ObjSuf), $(basename $(PROJECT_TEST_SRC)))
PROJECT_TEST_DEP = $(addsuffix .$(DepSuf), $(basename $(PROJECT_TEST_SRC)))


PROJECT_NEEDED_DIRECTORIES = $(addsuffix /$(OBJ_SUB_DIR), $(BASE_PROJECT_DIRS)) \
		$(addsuffix /$(BIN_SUB_DIR), $(BASE_PROJECT_DIRS)) \
		$(addsuffix /$(DEP_SUB_DIR), $(BASE_PROJECT_DIRS)) \
		$(addsuffix /$(INCL_SUB_DIR), $(BASE_PROJECT_DIRS)) \
		$(addsuffix /$(SRC_SUB_DIR), $(BASE_PROJECT_DIRS)) \
		$(addsuffix /$(LIB_SUB_DIR), $(BASE_PROJECT_DIRS))

COMP_OPTS = $(CXXFLAGS) $(ROOTINCL) $(CMSINCL) $(PROJ_INCLUDE)
LINK_OPTS = $(LDFLAGS) $(ROOTLIBS) $(CMSLIB)

#------------------------------------------------------------------------------

.PHONY : all library clean_deps deps clean test directory exe

all: $(PROJECT_NEEDED_DIRECTORIES) $(PROJECT_TEST_DEP) $(PROJECT_TEST_OBJ) $(PROJ_DEPS) $(PROJECT_OBJS) \
		$(PROJECT_TEST_BIN) library

directory: $(PROJECT_NEEDED_DIRECTORIES)	

library: $(MAIN_LIB_PATH)/libFit.$(DllSuf)
	
clean_deps:
	@rm -f $(PROJECT_DEPS_READY)


deps: $(PROJ_DEPS)

exe:
		$(CXX) $(LINK_OPTS) $(PROJECT_OBJ) $(MAIN_LIB_PATH)/libFit.$(DllSuf) -lCLHEP -o $(PROJECT_MAIN)

clean:
		@rm -f $(foreach dir, $(BASE_PROJECT_DIRS), $(wildcard $(dir)/$(OBJ_SUB_DIR)/*.$(ObjSuf)))
		@rm -f $(MAIN_BIN_PATH)/* $(MAIN_LIB_PATH)/*
		@rm -f $(PROJ_DEPS)
		@rm -f $(MAIN_SRC_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf)
		@rm -f $(MAIN_STUB_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf)
		@rm -f $(PROJECT_TEST_OBJ) $(PROJECT_TEST_DEP) $(PROJECT_TEST_BIN)
#------------------------------------------------------------------------------

$(PROJECT_NEEDED_DIRECTORIES):
	mkdir -p $@


$(MAIN_LIB_PATH)/libFit.$(DllSuf): $(PROJ_DEPS) $(PROJECT_OBJS)
	$(LD) $(SOFLAGS) $(LINK_OPTS) $(PROJECT_OBJS_READY) -o $(MAIN_LIB_PATH)/libFit.$(DllSuf)

$(MAIN_OBJ_PATH)/$(CINT_OUT_FILE_BASE).$(ObjSuf): $(MAIN_SRC_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf)

$(MAIN_DEP_PATH)/$(CINT_OUT_FILE_BASE).$(DepSuf): $(MAIN_SRC_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf)


%.$(DepSuf):
	@echo making deps rules for $@
	$(CXX) $(COMP_OPTS) -MMD -E -MT $@ -MT $(join $(patsubst %$(DEP_SUB_DIR)/, %$(OBJ_SUB_DIR)/, $(dir $@)), $(addsuffix .$(ObjSuf), $(basename $(notdir $@)))) $(join $(patsubst %$(DEP_SUB_DIR)/, %$(SRC_SUB_DIR)/, $(dir $@)), $(addsuffix .$(SrcSuf), $(basename $(notdir $@)))) -o $(addsuffix .$(ObjSuf), $(basename $@))
	@rm -f $(addsuffix .$(ObjSuf), $(basename $@))


%.$(ObjSuf): 
	@echo file to prepere: $@
	$(CXX) $(COMP_OPTS) -c $(join $(patsubst %$(OBJ_SUB_DIR)/, %$(SRC_SUB_DIR)/, $(dir $@)), $(addsuffix .$(SrcSuf), $(basename $(notdir $@)))) -o $@

$(MAIN_SRC_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf): $(MAIN_STUB_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf)
	@cp $(MAIN_STUB_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf) $(MAIN_SRC_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf)


$(MAIN_STUB_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf): $(MAIN_STUB_PATH)/$(CINT_CONF_FILE) $(PROJECT_STUB_HEADERS)
	@echo "Generating dictionary..."
	@echo rootcint -f $(MAIN_STUB_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf) -c $(ROOTINCL) $(CMSINCL) $(PROJ_INCLUDE) $(PROJECT_STUB_HEADERS) $(MAIN_STUB_PATH)/$(CINT_CONF_FILE)
	@rootcint -f $(MAIN_STUB_PATH)/$(CINT_OUT_FILE_BASE).$(SrcSuf) -c $(ROOTINCL) $(CMSINCL) $(PROJ_INCLUDE) $(PROJECT_STUB_HEADERS) $(MAIN_STUB_PATH)/$(CINT_CONF_FILE)


$(PROJECT_TEST_BIN): $(PROJECT_OBJS)
	$(LD) $(LINK_OPTS) $(PROJECT_OBJS_READY) $(join $(patsubst %$(BIN_SUB_DIR)/, %$(TEST_SUB_DIR)/, $(dir $@)), $(addsuffix .$(ObjSuf), $(notdir $@))) -o $@


$(PROJECT_TEST_OBJ): 
	@echo file to prepere: $@
	$(CXX) $(COMP_OPTS) -c $(addsuffix .$(SrcSuf), $(basename $@)) -o $@


$(PROJECT_TEST_DEP):
	@echo making deps rules for $@
	$(CXX) $(COMP_OPTS) -MMD -E -MT $@ -MT $(addsuffix .$(ObjSuf), $(basename $@)) $(addsuffix .$(SrcSuf), $(basename $@)) -MT $(join $(patsubst %$(TEST_SUB_DIR)/, %$(BIN_SUB_DIR)/, $(dir $@)), $(notdir $(basename $@))) -o $(addsuffix .$(ObjSuf), $(basename $@))
	@rm -f $(addsuffix .$(ObjSuf), $(basename $@))


-include $(PROJECT_DEPS_READY)
