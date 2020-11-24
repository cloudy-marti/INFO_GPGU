#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class cuStopwatch{
    // todo: add your internal data structure, all in private
	private:
		cudaEvent_t startEvent;
		cudaEvent_t endEvent;
		bool started;

    public:
        cuStopwatch();
        ~cuStopwatch();
        void start();
        float stop();
};

cuStopwatch::cuStopwatch(){
    // todo: constructor
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    started = false;
}

cuStopwatch::~cuStopwatch(){
    // todo: destructor
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);
}

void cuStopwatch::start(){
    // todo: start the stopwatch, and ignore double start
    if(!started) {
    	cudaEventRecord(startEvent);
    	started = true;
    }
}

float cuStopwatch::stop(){
    // todo: stop the stopwatch and return elapsed time, ignore invalid stops (e.g. stop when not yet started or double stop)
	if(!started) {
		return -1;
	}
	cudaEventSynchronize(startEvent);
	
	cudaEventRecord(endEvent);
	cudaEventSynchronize(endEvent);
	
	float ms;
	cudaEventElapsedTime(&ms, startEvent, endEvent);

	started = false;
	
	return ms;
}