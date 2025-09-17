
#include <cstdio>       // for printf
#include <string>       // for std::stoi

void print_array(int* arr, int size) {
    printf("Array elements: ");
    for (int i = 0; i < size; i++) {
        printf("%d, ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {

    // Check if the correct number of arguments are provided
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]);
        return 1;
    }

    // Convert the first argument to an integer
    int input_val = std::stoi(argv[1]);


    // Dynamically allocate memory for the array
    int* arr = (int*) malloc(10 * sizeof(int));
    if (arr ==nullptr) {
        printf("Error: memory allocation failed. \n");
        return 1;
    }

    // Fill the array with the specified values and print the array
    arr[0] = input_val;
    for (int i = 1; i <= 9; i++) {
        arr[i] = arr[i-1] + 1;
    }
    print_array(arr, 10);
    
    // Calculate the sum of array elements and print the sum
    int sum = 0;
    for (int i = 0; i<=9; i++) {
        sum+=arr[i];
    }
    printf("Sum: %d\n", sum);

    //Free allocated memory to avoid memory leaks
    free(arr);
    arr = nullptr; // To avoid dangling points, point array to null.

    return 0;
}
