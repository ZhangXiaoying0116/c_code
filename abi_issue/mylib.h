// mylib.h

#ifndef MYLIB_H
#define MYLIB_H


// old
// typedef struct {
//     int old_field;
// } mylib_mystruct;

// mylib_mystruct* mylib_init(int old_field);

// new
typedef struct {
    int new_field;
    int old_field;
} mylib_mystruct;

mylib_mystruct* mylib_init(int old_field);
#endif


