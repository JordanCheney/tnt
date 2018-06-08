function(tnt_add_sanitizers target)
  target_compile_options(${target} PRIVATE -fsanitize=memory -fno-omit-frame-pointer -g -O2)
endfunction()
