rem nvcc -o conv-filter.exe -O2 --library-path SDL\lib --include-path SDL\include --compiler-options "/EHsc /MD" --linker-options "/SUBSYSTEM:WINDOWS SDL2.lib SDL2main.lib SDL2_image.lib" conv-filter.cu

nvcc -o conv-filter.exe -O2 --library-path SDL\lib --include-path SDL\include --compiler-options "/EHsc /MD" --linker-options "/SUBSYSTEM:CONSOLE SDL2.lib SDL2main.lib SDL2_image.lib" conv-filter.cu
