#include <stdio.h>
#include <math.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


#define lsvalue = 0.34f

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call)                                      \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n",             \
                    cudaGetErrorString(err), __FILE__, __LINE__);   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0);

int gcd(int a, int b){
    while (b != 0) {
        int tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}
void RSAcore( int pubKey, int privKey, int * e, int * d){

    int n, tot_n, temp;
    n = pubKey * privKey;
    tot_n = (pubKey - 1) * (privKey - 1);


    int coun=1;
    for(int x = 12 ; x < tot_n ; x++){
        if(gcd(x,tot_n)==1){
            *e=x;
            break;
        }
    }

    for (int y = 10; y < tot_n;y++){
        if ((y * *e)%tot_n == 1){
            printf("%d \n",y);

        }
    }

}
int encrRSA(int message, int pubKey, int privKey){

    int cipher,e, d;

    RSAcore(pubKey, privKey, &e, &d);
    printf("%d ",e);


    return cipher;
}

int decrRSA(int cipher, int pubKey, int privKey){
    int message,e,d;
    RSAcore(pubKey, privKey, &e, &d);
    return message;
}


int main(){

    int pubKey,privKey;

    scanf("%d %d",&pubKey,&privKey);

    encrRSA(78,pubKey,privKey);


}