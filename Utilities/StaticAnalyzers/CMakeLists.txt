
cmake_minimum_required (VERSION 2.6)
project (clang_cms)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

# set your llvm/clang paths here
set (LLVM_SRC_DIR ../llvm)
set (LLVM_SRC_BUILD_DIR ../build)
set (LLVM_BIN_BUILD_DIR ${LLVM_SRC_BUILD_DIR}/Release+Asserts)


# the rest should work automatically

# assumes for clang to be under llvm/tools/, as suggested by the clang website 
# http://clang.llvm.org/get_started.html#build
set (CLANG_SRC_DIR  ${LLVM_SRC_DIR}/tools/clang)
set (CLANG_BUILD_DIR ${LLVM_SRC_BUILD_DIR}/tools/clang)

add_definitions (-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)
add_definitions (-D_GNU_SOURCE -DHAVE_CLANG_CONFIG_H)

# omit reports for some CMS specific cases
add_definitions ( -DTHREAD_CHECKS_USE_CMS_EXCEPTIONS )

# omit reports found in the system headers
add_definitions ( -DTHREAD_CHECKS_NO_REPORT_SYSTEM )


# this sets up the devel clang as our compiler
set (CMAKE_CXX_COMPILER "${LLVM_BIN_BUILD_DIR}/bin/clang++")
set (CMAKE_CC_COMPILER "${LLVM_BIN_BUILD_DIR}/bin/clang")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common -Woverloaded-virtual -Wcast-qual -fno-strict-aliasing -pedantic -Wno-long-long -Wall -W -Wno-unused-parameter -Wwrite-strings -fno-exceptions -fno-rtti")

set (CMAKE_MODULE_LINKER_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")

# not used because they will be available anyhow if the plugin is loaded into a clang host application
set (LLVM_LIBS
#	LLVMJIT
#	LLVMX86CodeGen
#	LLVMX86AsmParser
#	LLVMX86Disassembler
#	LLVMExecutionEngine
#	LLVMAsmPrinter
#	LLVMSelectionDAG
#	LLVMX86AsmPrinter
#	LLVMX86Info
#	LLVMMCParser
#	LLVMCodeGen
#	LLVMX86Utils
#	LLVMScalarOpts
#	LLVMInstCombine
#	LLVMTransformUtils
#	LLVMipa
#	LLVMAnalysis
#	LLVMTarget
#	LLVMCore
#	LLVMMC
#	LLVMSupport
)


macro(add_clang_plugin name)
	set (srcs ${ARGN})

	include_directories( "${LLVM_SRC_DIR}/include"
		"${LLVM_SRC_BUILD_DIR}/include"
		"${CLANG_SRC_DIR}/include"
		"${CLANG_BUILD_DIR}/include" )
	link_directories( "${LLVM_BUILD_DIR}/lib" )

	add_library( ${name} SHARED ${srcs} )

	if (SYMBOL_FILE)
		set_target_properties( ${name} PROPERTIES LINK_FlAGS
			"-exported_symbols_list ${SYMBOL_FILE}")
	endif()

	foreach (clang_lib ${CLANG_LIBS})
		target_link_libraries( ${name} ${clang_lib} )
	endforeach()

	foreach (llvm_lib ${LLVM_LIBS})
		target_link_libraries( ${name} ${llvm_lib} )
	endforeach()

	foreach (user_lib ${USER_LIBS})
		target_link_libraries( ${name} ${user_lib} )
	endforeach()

	set_target_properties( ${name} PROPERTIES
		LINKER_LANGUAGE CXX
		PREFIX "")
endmacro(add_clang_plugin)

# not used because they will be available anyhow if the plugin is loaded into a clang host application
set (CLANG_LIBS
#	clang
#	clangFrontend
#	clangAST
#	clangAnalysis
#	clangBasic
#	clangCodeGen
#	clangDriver
#	clangFrontendTool
#	clangLex
#	clangParse
#	clangRewrite
#	clangSema
#	clangSerialization
#	clangStaticAnalyzerCheckers
#	clangStaticAnalyzerCore
#	clangStaticAnalyzerFrontend
)

set (USER_LIBS
	pthread
)

add_clang_plugin( ClangCms 
	src/ClangCmsCheckerPluginRegister.cpp 
	src/CmsException.cpp
	src/ConstCastChecker.cpp 
	src/ConstCastAwayChecker.cpp
	src/StaticLocalChecker.cpp
	src/GlobalStaticChecker.cpp
	src/MutableMemberChecker.cpp)


