# Extract version info from header file
macro(tnt_get_version VAR)
  file(STRINGS ${PROJECT_SOURCE_DIR}/include/tnt/tnt.hpp LINE
         REGEX "#define[ ]+${VAR}[ ]+[0-9]+")
  string(REGEX MATCH "[0-9]+" ${VAR} ${LINE})
endmacro(tnt_get_version)

tnt_get_version(TNT_VERSION_MAJOR)
tnt_get_version(TNT_VERSION_MINOR)
tnt_get_version(TNT_VERSION_PATCH)

set(TNT_VERSION ${TNT_VERSION_MAJOR}.${TNT_VERSION_MINOR}.${TNT_VERSION_PATCH})
