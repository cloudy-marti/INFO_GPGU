#include <stdint.h>
#include <time.h>

void randgen(uint64_t* arr, size_t count){
    uint64_t state = time(NULL);
    state ^= state << 12;
    state += state >> 7;
    state ^= state << 23;
    state += state >> 6;
    state ^= state << 45;
    state -= state >> 4;
    state++;
    for(uint64_t i = 0; i < count; i++){
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        arr[i] = state;
    }
    return;
}
