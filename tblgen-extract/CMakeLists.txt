set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(tblgen-extract
  tblgen-extract.cpp
  )

llvm_update_compile_flags(tblgen-extract)
target_link_libraries(tblgen-extract
  PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  MLIRTableGen
  )

mlir_check_link_libraries(tblgen-extract)
