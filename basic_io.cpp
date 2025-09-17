
#include <cstdio>       // for printf
#include <string>       // for std::stoi

int main(int argc, char *argv[]) {

    // Check if the correct number of arguments are provided
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    // Convert the first argument to an integer
    int input_val = std::stoi(argv[1]);

    /* Part A: Basic Syntax and I/O */
    // Double the number
    int d_val = input_val*2;
    
    // Print original and new
    printf("Original: %d\n", input_val);
    printf("Doubled: %d\n", d_val);
    
    /* Part B: Control Structures */
    // Print the sign
    if (input_val < 0) {
        printf("Original is: negative\n");
    } else if (input_val > 0) {
        printf("Original is: positive\n");
    } else {
        printf("Original is: zero\n");
    }


    return 0;
}
