option(BUILD_TNT_DOCS OFF "Build TNT documentation")
if (BUILD_TNT_DOCS)
  find_package(standardese REQUIRED)
  standardese_generate(docs CONFIG standardese.config
                       INPUT include/tnt)
endif()
