rem nvcc -o mandelbrot.exe -O2 --library-path SDL\lib --include-path SDL\include --compiler-options "/EHsc /MD" --linker-options "/SUBSYSTEM:WINDOWS SDL2.lib SDL2main.lib" mandelbrot.cu
nvcc -o mandelbrot.exe -O2 --library-path SDL\lib --include-path SDL\include --compiler-options "/EHsc /MD" --linker-options "/SUBSYSTEM:CONSOLE SDL2.lib SDL2main.lib" mandelbrot.cu
