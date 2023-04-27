# linux编译c文件成动态库
gcc -fPIC -shared -o libmylib.so mylib.c

# 编译与链接动态库
gcc main.c -I`pwd` -L. -lmylib -o main

# 执行
./main