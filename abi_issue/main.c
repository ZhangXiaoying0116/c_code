// main.c
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "mylib.h"

int main(void) {
    mylib_mystruct *myobject = mylib_init(1);
    printf("myobject->old_field address =  %p\n", &(*myobject));
    printf("myobject->old_field address =  %p\n", &(myobject->old_field));
    // printf("myobject->new_field address =  %p\n", &(myobject->new_field));
    // printf("myobject->old_field address =  %p\n", &(myobject->old_field));
    // assert(myobject->old_field == 1);
    free(myobject);
    return EXIT_SUCCESS;
}