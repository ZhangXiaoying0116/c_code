// mylib.c

#include <stdlib.h>
#include <stdio.h>
#include "mylib.h"

mylib_mystruct* mylib_init(int old_field) {
    mylib_mystruct *myobject;
    myobject = malloc(sizeof(mylib_mystruct));
    myobject->old_field = old_field;
    printf("1myobject->old_field address =  %p\n", &(*myobject));
    printf("1myobject->old_field address =  %p\n", &(myobject->old_field));
    return myobject;
}